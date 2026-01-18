"""Warmup module for index pre-loading with generation-aware locking."""
from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .config import INDEX_DIR
from .index_ops import get_current_generation_pointer, get_generation_dir
from .utils import env_bool, env_int, is_primary_process, log_info, log_notice, log_warn, process_role

if TYPE_CHECKING:
    from .context import AppState

SYSTEM_VERSION = "1.3.0"

# Module-level state
_last_warmup_ts: float = 0.0
_last_warmup_mono: float = 0.0  # monotonic time for interval checks
_warmup_lock = threading.Lock()  # Short-term lock to prevent concurrent warmup


def _warmup_lock_path(gen_name: str) -> Path:
    """Get the lock file path for a generation."""
    return INDEX_DIR / f".warmup_{gen_name}.lock"


def _remove_warmup_lock(gen_name: str) -> bool:
    """Remove warmup lock file for a generation.

    Returns True if removed, False otherwise.
    """
    lock_path = _warmup_lock_path(gen_name)
    try:
        lock_path.unlink()
        return True
    except OSError:
        return False


def _try_acquire_generation_lock(gen_name: str) -> bool:
    """Attempt to acquire generation warmup lock atomically.

    Uses O_CREAT|O_EXCL for atomic file creation.
    Returns True if lock acquired, False if already exists.
    """
    lock_path = _warmup_lock_path(gen_name)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"{time.time()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except OSError as e:
        log_warn(f"warmupロック作成失敗: gen={gen_name} error={e}")
        return False


def _read_touch_file(
    path: Path,
    head_bytes: int,
    stride_bytes: int,
    page_bytes: int,
    max_bytes: int | None,
) -> int:
    """Read file to load into OS page cache."""
    bytes_read = 0
    try:
        size = path.stat().st_size
    except Exception:
        return 0
    if size <= 0:
        return 0
    try:
        with path.open("rb") as f:
            if head_bytes > 0:
                head = f.read(min(head_bytes, size))
                bytes_read += len(head)
            if stride_bytes > 0 and size > head_bytes:
                offset = head_bytes
                while offset < size:
                    if max_bytes is not None and bytes_read >= max_bytes:
                        break
                    f.seek(offset)
                    chunk = f.read(page_bytes)
                    bytes_read += len(chunk)
                    offset += stride_bytes
    except Exception:
        return bytes_read
    return bytes_read


def run_warmup_once(reason: str, gen_name: str | None = None) -> bool:
    """Run warmup for a generation.

    For startup/generation_switch: runs exactly once per generation (file lock).
    For keep-warm: runs periodically when idle (short-term thread lock only).

    Args:
        reason: Reason for warmup (startup, generation_switch, keep-warm)
        gen_name: Generation name (defaults to current generation)

    Returns:
        True if warmup was executed, False if skipped
    """
    global _last_warmup_ts, _last_warmup_mono

    role = process_role()
    if not is_primary_process():
        log_notice(f"warmupスキップ: role={role} reason={reason}")
        return False

    # Get current generation if not specified
    if gen_name is None:
        gen_name = get_current_generation_pointer()
    if not gen_name:
        log_warn(f"warmupスキップ: 世代が見つかりません reason={reason}")
        return False

    # Check if warmup is enabled
    if not env_bool("WARMUP_ENABLED", True):
        log_notice(f"warmupスキップ: disabled reason={reason} gen={gen_name}")
        return False

    # Get generation directory BEFORE acquiring lock
    gen_dir = get_generation_dir(gen_name)
    if not gen_dir.exists():
        log_warn(f"warmupスキップ: gen_{gen_name} が見つかりません reason={reason}")
        return False

    # For startup/generation_switch: use generation lock (once per generation)
    # For keep-warm: skip generation lock (run periodically)
    use_generation_lock = reason != "keep-warm"

    # Acquire short-term thread lock first to prevent concurrent warmup
    # (including concurrent keep-warm and generation_switch)
    if not _warmup_lock.acquire(blocking=False):
        log_notice(f"warmupスキップ: 既に実行中 reason={reason}")
        return False

    try:
        if use_generation_lock:
            # Try to acquire generation lock (atomic, persistent)
            if not _try_acquire_generation_lock(gen_name):
                log_notice(f"warmupスキップ: 世代 {gen_name} は既に実行済み reason={reason}")
                return False

            # Double-check gen_dir still exists after lock acquisition
            if not gen_dir.exists():
                log_warn(f"warmup失敗: gen_{gen_name} がロック取得後に消失")
                _remove_warmup_lock(gen_name)
                return False

        # Get warmup parameters from environment
        head_mb = env_int("WARMUP_HEAD_MB", 2)
        stride_mb = env_int("WARMUP_STRIDE_MB", 4)
        max_mb = env_int("WARMUP_MAX_MB", 0)
        max_files = env_int("WARMUP_MAX_FILES", 40)

        # Collect files to warm
        files = sorted([p for p in gen_dir.iterdir() if p.is_file()])
        if not files:
            log_notice(f"warmup対象なし: gen_{gen_name}")
            _last_warmup_ts = time.time()
            _last_warmup_mono = time.monotonic()
            return True

        # Limit number of files if specified
        if max_files > 0:
            files = files[:max_files]

        max_bytes = max_mb * 1024 * 1024 if max_mb > 0 else None
        head_bytes = head_mb * 1024 * 1024
        stride_bytes = stride_mb * 1024 * 1024
        page_bytes = 4096

        log_info(
            f"warmup開始: reason={reason} role={role} gen={gen_name} "
            f"files={len(files)} head={head_mb}MB stride={stride_mb}MB max={max_mb}MB"
        )

        start = time.time()
        total_bytes = 0
        touched_files = 0

        for path in files:
            if max_bytes is not None and total_bytes >= max_bytes:
                break
            bytes_read = _read_touch_file(
                path,
                head_bytes,
                stride_bytes,
                page_bytes,
                max_bytes - total_bytes if max_bytes is not None else None,
            )
            if bytes_read:
                touched_files += 1
                total_bytes += bytes_read

        elapsed = time.time() - start
        _last_warmup_ts = time.time()
        _last_warmup_mono = time.monotonic()

        log_info(
            f"warmup完了: reason={reason} role={role} gen={gen_name} "
            f"files={touched_files}/{len(files)} bytes={total_bytes} elapsed={elapsed:.2f}s"
        )
        return True
    finally:
        _warmup_lock.release()


async def trigger_warmup(reason: str, gen_name: str | None = None) -> bool:
    """Async wrapper to run warmup in thread executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, run_warmup_once, reason, gen_name)


def should_run_keep_warm(
    last_search_ts: float,
    last_warmup_mono: float,
    active_client_count: int,
    idle_sec: int,
    interval_sec: int,
    now_ts: float | None = None,
    now_mono: float | None = None,
) -> bool:
    """Determine if keep-warm should run based on state.

    Args:
        last_search_ts: Timestamp of last search (time.time())
        last_warmup_mono: Monotonic time of last warmup (time.monotonic())
        active_client_count: Number of active clients
        idle_sec: Required idle time in seconds
        interval_sec: Minimum interval between warmups
        now_ts: Current time.time() (optional, for testing)
        now_mono: Current time.monotonic() (optional, for testing)

    Returns:
        True if keep-warm should run
    """
    if now_ts is None:
        now_ts = time.time()
    if now_mono is None:
        now_mono = time.monotonic()

    # Check if idle long enough (use wall clock for search timestamp)
    if now_ts - last_search_ts < idle_sec:
        return False

    # Check if interval has passed since last warmup (use monotonic to avoid clock skew)
    if now_mono - last_warmup_mono < interval_sec:
        return False

    # Check if no active clients
    if active_client_count > 0:
        return False

    return True


async def keep_warm_loop(state: AppState) -> None:
    """Background task to keep indexes warm when idle.

    Args:
        state: Application state containing last_search_ts, active_client_heartbeats, etc.
    """
    if not env_bool("WARMUP_ENABLED", True):
        return
    if not is_primary_process():
        return

    # Get configuration
    idle_sec = env_int("WARMUP_IDLE_SEC", 1800)  # 30 minutes default
    interval_sec = env_int("WARMUP_INTERVAL_SEC", 3600)  # 1 hour default
    # Ensure check_interval is at least 1 second to avoid busy loop
    check_interval = max(1, min(60, idle_sec // 10))

    role = process_role()
    log_info(f"keep-warmループ開始: role={role} idle={idle_sec}s interval={interval_sec}s check={check_interval}s")

    try:
        while True:
            await asyncio.sleep(check_interval)

            # Get active client count (thread-safe)
            with state.active_client_lock:
                ttl_sec = env_int("HEARTBEAT_TTL_SEC", 90)
                now_ts = time.time()
                active_count = sum(
                    1 for ts in state.active_client_heartbeats.values()
                    if now_ts - ts < ttl_sec
                )

            # Get last search timestamp from state (wall clock)
            last_search = getattr(state, 'last_search_ts', 0.0)

            # Check if we should run keep-warm (use monotonic time for warmup interval)
            if not should_run_keep_warm(
                last_search, _last_warmup_mono, active_count, idle_sec, interval_sec
            ):
                continue

            # Run warmup for current generation
            gen_name = get_current_generation_pointer()
            if gen_name:
                idle_time = time.time() - last_search
                executed = await trigger_warmup("keep-warm", gen_name)
                if executed:
                    log_notice(f"keep-warm完了: idle={idle_time:.0f}s active_clients={active_count}")
                    # Update state timestamp (wall clock for external use)
                    if hasattr(state, 'last_warmup_ts'):
                        state.last_warmup_ts = time.time()
    except asyncio.CancelledError:
        log_info("keep-warmループ終了")
        raise


def get_last_warmup_ts() -> float:
    """Get the timestamp of last warmup."""
    return _last_warmup_ts


def cleanup_old_warmup_locks(keep_generations: list[str] | None = None) -> int:
    """Remove warmup lock files for non-existent generations.

    Args:
        keep_generations: List of generation names to keep (optional)

    Returns:
        Number of lock files removed
    """
    removed = 0
    try:
        for lock_file in INDEX_DIR.glob(".warmup_*.lock"):
            # Extract generation name from lock file
            gen_name = lock_file.name[8:-5]  # Remove ".warmup_" prefix and ".lock" suffix

            # Check if generation directory exists
            gen_dir = INDEX_DIR / f"gen_{gen_name}"
            if not gen_dir.exists():
                try:
                    lock_file.unlink()
                    removed += 1
                    log_notice(f"warmupロック削除: {lock_file.name} (世代不在)")
                except OSError:
                    pass
            elif keep_generations is not None and gen_name not in keep_generations:
                try:
                    lock_file.unlink()
                    removed += 1
                    log_notice(f"warmupロック削除: {lock_file.name} (保持対象外)")
                except OSError:
                    pass
    except Exception as e:
        log_warn(f"warmupロッククリーンアップ失敗: {e}")
    return removed
