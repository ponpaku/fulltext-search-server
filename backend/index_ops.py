"""Index operations: generation management, file state, and index building."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from .config import (
    ALLOWED_EXTS,
    BUILD_DIR,
    CURRENT_POINTER_FILE,
    INDEX_DIR,
    INDEX_VERSION,
)
from .extractors import extract_text_from_file_with_reason
from .utils import env_bool, env_int, log_info, log_notice, log_warn

SYSTEM_VERSION = "1.1.11"
# File Version: 1.0.0


def create_generation_uuid() -> str:
    """Generate a new generation UUID (timestamp + random)."""
    import uuid
    timestamp = int(time.time())
    random_part = uuid.uuid4().hex[:8]
    return f"{timestamp}_{random_part}"


def get_current_generation_pointer() -> str | None:
    """Get generation name from current pointer file."""
    if not CURRENT_POINTER_FILE.exists():
        return None
    try:
        gen_name = CURRENT_POINTER_FILE.read_text(encoding="utf-8").strip()
        return gen_name if gen_name else None
    except Exception:
        return None


def set_current_generation_pointer(gen_name: str) -> None:
    """Set generation name to current pointer file (atomic)."""
    temp_file = CURRENT_POINTER_FILE.with_suffix(".tmp")
    try:
        temp_file.write_text(gen_name, encoding="utf-8")
        temp_file.replace(CURRENT_POINTER_FILE)
    except Exception as e:
        log_warn(f"currentポインター設定失敗: {e}")
        if temp_file.exists():
            temp_file.unlink()


def get_generation_dir(gen_name: str | None = None, build: bool = False) -> Path:
    """Get path to generation directory."""
    if gen_name is None:
        gen_name = get_current_generation_pointer()
        if gen_name is None:
            return INDEX_DIR
    base = BUILD_DIR if build else INDEX_DIR
    return base / f"gen_{gen_name}"


def get_current_generation_dir() -> Path:
    """Get current generation directory (auto-select latest if current.txt missing)."""
    gen_name = get_current_generation_pointer()
    if gen_name:
        gen_dir = get_generation_dir(gen_name, build=False)
        if gen_dir.exists():
            return gen_dir

    generations = list_generations()
    if generations:
        latest_gen_name, latest_gen_dir, latest_manifest = generations[0]
        log_warn(f"current.txt が見つからないため、最新世代を使用: gen_{latest_gen_name}")
        set_current_generation_pointer(latest_gen_name)
        return latest_gen_dir

    return INDEX_DIR


def create_manifest(gen_name: str, gen_dir: Path, folder_states_snapshot: Dict) -> Dict:
    """Create manifest.json."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)

    folder_paths = sorted([meta["path"] for meta in folder_states_snapshot.values()])
    source_folders_hash = hashlib.sha256(
        "|".join(folder_paths).encode("utf-8")
    ).hexdigest()

    folders_info = {}
    for folder_id, meta in folder_states_snapshot.items():
        path = meta["path"]
        hashed = hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
        index_file = f"index_{hashed}_{INDEX_VERSION}.pkl.gz"
        index_path = gen_dir / index_file
        file_count = 0
        if index_path.exists():
            try:
                with gzip.open(index_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        file_count = len(data)
            except Exception:
                pass

        folders_info[folder_id] = {
            "path": path,
            "name": meta["name"],
            "file_count": file_count,
            "index_file": index_file,
        }

    manifest = {
        "index_uuid": gen_name,
        "schema_version": INDEX_VERSION,
        "created_at": now.isoformat(),
        "created_timestamp": int(now.timestamp()),
        "source_folders_hash": source_folders_hash,
        "folders": folders_info,
    }

    return manifest


def save_manifest(gen_dir: Path, manifest: Dict) -> None:
    """Save manifest.json."""
    manifest_path = gen_dir / "manifest.json"
    try:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_warn(f"manifest.json 保存失敗: {e}")


def load_manifest(gen_dir: Path) -> Dict | None:
    """Load manifest.json."""
    manifest_path = gen_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_current_generation_manifest() -> Dict | None:
    """Get manifest.json for current generation."""
    try:
        gen_dir = get_current_generation_dir()
        if not gen_dir or not gen_dir.exists():
            return None
        return load_manifest(gen_dir)
    except Exception:
        return None


def list_generations() -> List[Tuple[str, Path, Dict | None]]:
    """List all generations (gen_name, path, manifest)."""
    generations = []
    for item in INDEX_DIR.iterdir():
        if item.is_dir() and item.name.startswith("gen_"):
            gen_name = item.name[4:]
            manifest = load_manifest(item)
            generations.append((gen_name, item, manifest))

    def get_sort_key(item):
        gen_name, gen_dir, manifest = item
        if manifest and "created_timestamp" in manifest:
            return manifest["created_timestamp"]
        try:
            timestamp_str = gen_name.split("_")[0]
            return int(timestamp_str)
        except (ValueError, IndexError):
            return 0

    generations.sort(key=get_sort_key, reverse=True)
    return generations


def cleanup_old_generations(current_gen_name: str | None, grace_sec: int = 300) -> None:
    """Delete old generations according to retention policy."""
    import shutil

    keep_generations = env_int("INDEX_KEEP_GENERATIONS", 3)
    keep_days = env_int("INDEX_KEEP_DAYS", 0)
    max_bytes = env_int("INDEX_MAX_BYTES", 0)

    all_generations = list_generations()

    current_gen_dir = None
    if current_gen_name:
        current_gen_dir = get_generation_dir(current_gen_name, build=False)

    eligible_generations = []
    for gen_name, gen_dir, manifest in all_generations:
        if current_gen_dir and gen_dir == current_gen_dir:
            continue

        if manifest and grace_sec > 0:
            created_timestamp = manifest.get("created_timestamp", 0)
            age_sec = time.time() - created_timestamp
            if age_sec < grace_sec:
                continue

        eligible_generations.append((gen_name, gen_dir, manifest))

    to_delete = []

    if keep_generations > 0 and len(eligible_generations) > keep_generations:
        for gen_name, gen_dir, manifest in eligible_generations[keep_generations:]:
            to_delete.append((gen_name, gen_dir, "世代数超過"))

    if keep_days > 0:
        for gen_name, gen_dir, manifest in eligible_generations:
            if (gen_name, gen_dir, "世代数超過") in to_delete:
                continue
            if manifest:
                created_timestamp = manifest.get("created_timestamp", 0)
                age_days = (time.time() - created_timestamp) / 86400
                if age_days > keep_days:
                    to_delete.append((gen_name, gen_dir, "保持期限切れ"))

    if max_bytes > 0:
        total_bytes = 0
        keep_set = set((gen_name, gen_dir) for gen_name, gen_dir, _ in to_delete)

        if current_gen_dir and current_gen_dir.exists():
            try:
                current_bytes = sum(
                    f.stat().st_size for f in current_gen_dir.rglob("*") if f.is_file()
                )
                total_bytes += current_bytes
            except Exception:
                pass

        for gen_name, gen_dir, manifest in all_generations:
            if current_gen_dir and gen_dir == current_gen_dir:
                continue
            if (gen_name, gen_dir) in keep_set:
                continue
            try:
                dir_bytes = sum(
                    f.stat().st_size for f in gen_dir.rglob("*") if f.is_file()
                )
                total_bytes += dir_bytes
            except Exception:
                pass

        if total_bytes > max_bytes:
            for gen_name, gen_dir, manifest in reversed(eligible_generations):
                if (gen_name, gen_dir) in keep_set:
                    continue
                try:
                    dir_bytes = sum(
                        f.stat().st_size for f in gen_dir.rglob("*") if f.is_file()
                    )
                    total_bytes -= dir_bytes
                    to_delete.append((gen_name, gen_dir, "容量超過"))
                    if total_bytes <= max_bytes:
                        break
                except Exception:
                    pass

    for gen_name, gen_dir, reason in to_delete:
        try:
            shutil.rmtree(gen_dir)
            log_info(f"世代削除: gen_{gen_name} 理由={reason}")
        except Exception as e:
            log_warn(f"世代削除失敗: gen_{gen_name} エラー={e}")


def index_path_for(folder_path: str, gen_dir: Path | None = None) -> Path:
    """Get path to index file (generation directory aware)."""
    hashed = hashlib.sha256(folder_path.encode("utf-8")).hexdigest()[:12]
    if gen_dir is None:
        gen_dir = get_current_generation_dir()
    return gen_dir / f"index_{hashed}_{INDEX_VERSION}.pkl.gz"


def file_state_path_for(folder_path: str, gen_dir: Path | None = None) -> Path:
    """Get path to file_state.jsonl (generation directory aware)."""
    hashed = hashlib.sha256(folder_path.encode("utf-8")).hexdigest()[:12]
    if gen_dir is None:
        gen_dir = get_current_generation_dir()
    return gen_dir / f"file_state_{hashed}.jsonl"


def failures_path_for(folder_path: str, gen_dir: Path | None = None) -> Path:
    """Get path to failures.json (generation directory aware)."""
    hashed = hashlib.sha256(folder_path.encode("utf-8")).hexdigest()[:12]
    if gen_dir is None:
        gen_dir = get_current_generation_dir()
    return gen_dir / f"failures_{hashed}.json"


def load_file_state(folder_path: str, gen_dir: Path | None = None) -> Dict[str, Dict]:
    """Load file_state.jsonl from disk (generation directory aware)."""
    state_path = file_state_path_for(folder_path, gen_dir)
    if not state_path.exists():
        return {}
    states = {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if "path" not in entry:
                        log_warn(f"file_state.jsonl 読み込み警告: 行{line_num} に 'path' キーがありません")
                        continue
                    states[entry["path"]] = entry
                except json.JSONDecodeError:
                    log_warn(f"file_state.jsonl 読み込み警告: 行{line_num} の JSON パースに失敗しました")
                    continue
                except Exception as e:
                    log_warn(f"file_state.jsonl 読み込み警告: 行{line_num} の処理中にエラー ({e})")
                    continue
    except Exception as e:
        log_warn(f"file_state.jsonl 読み込みエラー: {state_path} ({e})")
    return states


def save_file_state(folder_path: str, states: Dict[str, Dict], gen_dir: Path | None = None):
    """Save file_state.jsonl to disk (generation directory aware)."""
    state_path = file_state_path_for(folder_path, gen_dir)
    if gen_dir is not None:
        gen_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            for path_str in sorted(states.keys()):
                f.write(json.dumps(states[path_str], ensure_ascii=False) + "\n")
    except Exception:
        return


def load_failures(folder_path: str, gen_dir: Path | None = None) -> Tuple[Dict[str, str], bool]:
    """Load failures.json from disk (generation directory aware)."""
    failures_path = failures_path_for(folder_path, gen_dir)
    if not failures_path.exists():
        return {}, False
    try:
        with open(failures_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}, True
    except Exception as e:
        log_warn(f"failures_*.json 読み込みエラー: {failures_path} ({e})")
    return {}, False


def save_failures(folder_path: str, failures: Dict[str, str], gen_dir: Path | None = None):
    """Save failures.json to disk (generation directory aware)."""
    failures_path = failures_path_for(folder_path, gen_dir)
    if gen_dir is not None:
        gen_dir.mkdir(parents=True, exist_ok=True)
    temp_path = failures_path.with_suffix(".json.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False)
        os.replace(temp_path, failures_path)
    except Exception as e:
        log_warn(f"failures_*.json 保存失敗: {failures_path} ({e})")
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return


def get_file_stat(path_str: str) -> Dict:
    """Get file stat info (size, mtime_ns, inode/file_id)."""
    try:
        stat_info = os.stat(path_str)
        result = {
            "size": stat_info.st_size,
            "mtime_ns": stat_info.st_mtime_ns,
        }
        if hasattr(stat_info, "st_ino") and stat_info.st_ino != 0:
            result["inode"] = stat_info.st_ino
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes
                kernel32 = ctypes.windll.kernel32
                GENERIC_READ = 0x80000000
                FILE_SHARE_READ = 0x00000001
                OPEN_EXISTING = 3
                FILE_FLAG_BACKUP_SEMANTICS = 0x02000000

                handle = kernel32.CreateFileW(
                    path_str,
                    GENERIC_READ,
                    FILE_SHARE_READ,
                    None,
                    OPEN_EXISTING,
                    FILE_FLAG_BACKUP_SEMANTICS,
                    None
                )
                if handle != -1:
                    class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
                        _fields_ = [
                            ("dwFileAttributes", wintypes.DWORD),
                            ("ftCreationTime", wintypes.FILETIME),
                            ("ftLastAccessTime", wintypes.FILETIME),
                            ("ftLastWriteTime", wintypes.FILETIME),
                            ("dwVolumeSerialNumber", wintypes.DWORD),
                            ("nFileSizeHigh", wintypes.DWORD),
                            ("nFileSizeLow", wintypes.DWORD),
                            ("nNumberOfLinks", wintypes.DWORD),
                            ("nFileIndexHigh", wintypes.DWORD),
                            ("nFileIndexLow", wintypes.DWORD),
                        ]
                    info = BY_HANDLE_FILE_INFORMATION()
                    if kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):
                        file_id = (info.nFileIndexHigh << 32) | info.nFileIndexLow
                        result["file_id"] = file_id
                    kernel32.CloseHandle(handle)
            except Exception:
                pass
        return result
    except Exception:
        return {}


def compute_fast_fingerprint(path_str: str, chunk_bytes: int = 65536) -> str | None:
    """Compute fast fingerprint (head/tail N bytes + size)."""
    try:
        stat_info = os.stat(path_str)
        file_size = stat_info.st_size

        hasher = hashlib.sha256()
        hasher.update(str(file_size).encode())

        with open(path_str, "rb") as f:
            head_chunk = f.read(chunk_bytes)
            hasher.update(head_chunk)

            if file_size > chunk_bytes:
                f.seek(-min(chunk_bytes, file_size - len(head_chunk)), 2)
                tail_chunk = f.read(chunk_bytes)
                hasher.update(tail_chunk)

        return hasher.hexdigest()
    except Exception:
        return None


def compute_full_hash(path_str: str, algo: str = "sha256") -> str | None:
    """Compute full hash."""
    try:
        hasher = hashlib.new(algo)
        with open(path_str, "rb") as f:
            while chunk := f.read(1048576):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def should_compute_full_hash(path_str: str) -> bool:
    """Determine if full hash should be computed."""
    full_hash_paths = os.getenv("FULL_HASH_PATHS", "").strip()
    if full_hash_paths:
        for prefix in full_hash_paths.split(","):
            prefix = prefix.strip()
            if prefix and path_str.startswith(prefix):
                return True

    full_hash_exts = os.getenv("FULL_HASH_EXTS", "").strip()
    if full_hash_exts:
        ext = os.path.splitext(path_str)[1].lower()
        for target_ext in full_hash_exts.split(","):
            target_ext = target_ext.strip().lower()
            if target_ext and ext == target_ext:
                return True

    if os.name == "nt":
        if path_str.startswith("\\\\") or path_str.startswith("//"):
            return True
    else:
        mount_prefixes = ["/mnt/", "/net/", "/Volumes/"]
        for prefix in mount_prefixes:
            if path_str.startswith(prefix):
                return True

    return False


def should_reindex_file(
    path_str: str,
    current_stat: Dict,
    prev_state: Dict | None,
    diff_mode: str,
    fast_fp_bytes: int,
    full_hash_algo: str,
) -> Tuple[bool, Dict]:
    """Perform staged diff check and return (should_reindex, updated_state)."""
    if not prev_state:
        new_state = {
            "path": path_str,
            "size": current_stat.get("size"),
            "mtime_ns": current_stat.get("mtime_ns"),
            "last_indexed_at": time.time(),
            "deletion_candidate_since": None,
        }
        if "inode" in current_stat:
            new_state["inode"] = current_stat["inode"]
        if "file_id" in current_stat:
            new_state["file_id"] = current_stat["file_id"]
        return True, new_state

    stat_changed = False
    if current_stat.get("size") != prev_state.get("size"):
        stat_changed = True
    elif current_stat.get("mtime_ns") != prev_state.get("mtime_ns"):
        stat_changed = True
    elif "inode" in current_stat and "inode" in prev_state:
        if current_stat["inode"] != prev_state["inode"]:
            stat_changed = True
    elif "file_id" in current_stat and "file_id" in prev_state:
        if current_stat["file_id"] != prev_state["file_id"]:
            stat_changed = True

    if not stat_changed:
        updated_state = prev_state.copy()
        updated_state["deletion_candidate_since"] = None
        return False, updated_state

    if diff_mode == "stat":
        new_state = {
            "path": path_str,
            "size": current_stat.get("size"),
            "mtime_ns": current_stat.get("mtime_ns"),
            "last_indexed_at": time.time(),
            "deletion_candidate_since": None,
        }
        if "inode" in current_stat:
            new_state["inode"] = current_stat["inode"]
        if "file_id" in current_stat:
            new_state["file_id"] = current_stat["file_id"]
        return True, new_state

    if diff_mode in {"stat+fastfp", "stat+fastfp+fullhash"}:
        current_fp = compute_fast_fingerprint(path_str, fast_fp_bytes)
        prev_fp = prev_state.get("fast_fp")

        if current_fp and prev_fp and current_fp == prev_fp:
            updated_state = prev_state.copy()
            updated_state["size"] = current_stat.get("size")
            updated_state["mtime_ns"] = current_stat.get("mtime_ns")
            if "inode" in current_stat:
                updated_state["inode"] = current_stat["inode"]
            if "file_id" in current_stat:
                updated_state["file_id"] = current_stat["file_id"]
            updated_state["deletion_candidate_since"] = None
            return False, updated_state

        if diff_mode == "stat+fastfp":
            new_state = {
                "path": path_str,
                "size": current_stat.get("size"),
                "mtime_ns": current_stat.get("mtime_ns"),
                "fast_fp": current_fp,
                "last_indexed_at": time.time(),
                "deletion_candidate_since": None,
            }
            if "inode" in current_stat:
                new_state["inode"] = current_stat["inode"]
            if "file_id" in current_stat:
                new_state["file_id"] = current_stat["file_id"]
            return True, new_state

    if diff_mode == "stat+fastfp+fullhash" and should_compute_full_hash(path_str):
        current_hash = compute_full_hash(path_str, full_hash_algo)
        prev_hash = prev_state.get("full_hash")

        if current_hash and prev_hash and current_hash == prev_hash:
            updated_state = prev_state.copy()
            updated_state["size"] = current_stat.get("size")
            updated_state["mtime_ns"] = current_stat.get("mtime_ns")
            if "inode" in current_stat:
                updated_state["inode"] = current_stat["inode"]
            if "file_id" in current_stat:
                updated_state["file_id"] = current_stat["file_id"]
            if diff_mode in {"stat+fastfp", "stat+fastfp+fullhash"}:
                current_fp = compute_fast_fingerprint(path_str, fast_fp_bytes)
                if current_fp:
                    updated_state["fast_fp"] = current_fp
            updated_state["deletion_candidate_since"] = None
            return False, updated_state

        new_state = {
            "path": path_str,
            "size": current_stat.get("size"),
            "mtime_ns": current_stat.get("mtime_ns"),
            "last_indexed_at": time.time(),
            "deletion_candidate_since": None,
        }
        if "inode" in current_stat:
            new_state["inode"] = current_stat["inode"]
        if "file_id" in current_stat:
            new_state["file_id"] = current_stat["file_id"]
        if diff_mode in {"stat+fastfp", "stat+fastfp+fullhash"}:
            current_fp = compute_fast_fingerprint(path_str, fast_fp_bytes)
            if current_fp:
                new_state["fast_fp"] = current_fp
        if current_hash:
            new_state["full_hash"] = current_hash
        return True, new_state

    new_state = {
        "path": path_str,
        "size": current_stat.get("size"),
        "mtime_ns": current_stat.get("mtime_ns"),
        "last_indexed_at": time.time(),
        "deletion_candidate_since": None,
    }
    if "inode" in current_stat:
        new_state["inode"] = current_stat["inode"]
    if "file_id" in current_stat:
        new_state["file_id"] = current_stat["file_id"]
    if diff_mode in {"stat+fastfp", "stat+fastfp+fullhash"}:
        current_fp = compute_fast_fingerprint(path_str, fast_fp_bytes)
        if current_fp:
            new_state["fast_fp"] = current_fp
    return True, new_state


def load_index_from_disk(folder_path: str, gen_dir: Path | None = None) -> Dict[str, Dict]:
    """Load index from disk (generation directory aware)."""
    idx_path = index_path_for(folder_path, gen_dir)
    if not idx_path.exists():
        return {}
    try:
        with gzip.open(idx_path, "rb") as f:
            data = pickle.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_index_to_disk(folder_path: str, cache: Dict[str, Dict], gen_dir: Path | None = None):
    """Save index to disk (generation directory aware)."""
    idx_path = index_path_for(folder_path, gen_dir)
    if gen_dir is not None:
        gen_dir.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(idx_path, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        return


def scan_files(folder: str) -> List[Path]:
    root = Path(folder)
    files = []
    try:
        for f in root.rglob("*"):
            try:
                if f.is_file() and f.suffix.lower() in ALLOWED_EXTS:
                    files.append(f)
            except (PermissionError, OSError):
                # Skip files that can't be accessed
                continue
    except (PermissionError, OSError) as e:
        # Log and continue with partial results if rglob itself fails
        from .utils import log_warn
        log_warn(f"scan_files: partial scan due to {type(e).__name__}: {e}")
    return files


def build_index_for_folder(
    folder: str,
    previous_failures: Dict[str, str] | None = None,
    gen_dir: Path | None = None,
    prev_gen_dir: Path | None = None,
    failures_loaded: bool | None = None,
) -> Tuple[Dict[str, Dict], Dict, Dict[str, str]]:
    """Build index with incremental updates and high-accuracy mode."""
    start_time = time.time()

    diff_mode = os.getenv("DIFF_MODE", "stat").strip().lower()
    if diff_mode not in {"stat", "stat+fastfp", "stat+fastfp+fullhash"}:
        diff_mode = "stat"
    try:
        fast_fp_bytes = int(os.getenv("FAST_FP_BYTES", "65536").strip())
        if fast_fp_bytes <= 0:
            fast_fp_bytes = 65536
    except (ValueError, AttributeError):
        fast_fp_bytes = 65536
    full_hash_algo = os.getenv("FULL_HASH_ALGO", "sha256").strip()
    if not full_hash_algo:
        full_hash_algo = "sha256"

    if prev_gen_dir is not None and prev_gen_dir.exists():
        existing_cache = load_index_from_disk(folder, prev_gen_dir)
        prev_file_states = load_file_state(folder, prev_gen_dir)
    else:
        existing_cache = load_index_from_disk(folder, gen_dir)
        prev_file_states = load_file_state(folder, gen_dir)
    failures = {k: v for k, v in (previous_failures or {}).items()}
    all_files = scan_files(folder)
    if not all_files and existing_cache:
        stats = {
            "total_files": len(existing_cache),
            "indexed_files": len(existing_cache),
            "updated_files": 0,
            "skipped_files": len(existing_cache),
        }
        elapsed = time.time() - start_time
        log_info(
            f"インデックス構築スキップ: {folder} 件数={stats['indexed_files']} "
            f"理由=ファイル一覧取得失敗 時間={elapsed:.1f}s"
        )
        failures = {k: v for k, v in failures.items() if k in existing_cache}
        return existing_cache, stats, failures
    if existing_cache and all_files:
        allow_shrink = os.getenv("REBUILD_ALLOW_SHRINK", "").strip().lower() in {"1", "true", "yes"}
        if not allow_shrink and len(all_files) < max(10, int(len(existing_cache) * 0.8)):
            stats = {
                "total_files": len(existing_cache),
                "indexed_files": len(existing_cache),
                "updated_files": 0,
                "skipped_files": len(existing_cache),
            }
            elapsed = time.time() - start_time
            log_info(
                f"インデックス構築スキップ: {folder} 件数={stats['indexed_files']} "
                f"理由=ファイル数急減 時間={elapsed:.1f}s"
            )
            failures = {k: v for k, v in failures.items() if k in existing_cache}
            return existing_cache, stats, failures

    current_map = {str(f): f for f in all_files}
    failures = {k: v for k, v in failures.items() if k in current_map}
    retry_set = set(failures.keys())
    if prev_gen_dir is not None and failures_loaded is False:
        bootstrap_set = {
            path_str
            for path_str in current_map
            if path_str in prev_file_states and path_str not in existing_cache
        }
        log_notice(
            "失敗履歴が無い/読み込めないため再試行対象を拡張: "
            f"{folder} 件数={len(bootstrap_set)} "
            f"(prev_states={len(prev_file_states)} cache={len(existing_cache)} files={len(current_map)})"
        )
        if bootstrap_set and not existing_cache and len(bootstrap_set) == len(current_map):
            log_warn(f"再試行対象が全件になります: {folder}")
        retry_set.update(bootstrap_set)

    valid_cache = dict(existing_cache)

    current_file_states: Dict[str, Dict] = {}
    for path_str in prev_file_states:
        if path_str not in current_map:
            prev_state = prev_file_states[path_str]
            if prev_state.get("deletion_candidate_since"):
                if path_str in valid_cache:
                    del valid_cache[path_str]
                    log_warn(f"削除確定（インデックス除外）: {path_str}")
            else:
                prev_state["deletion_candidate_since"] = time.time()
                current_file_states[path_str] = prev_state
                log_notice(f"削除候補: {path_str}")

    targets = []
    target_set = set()

    def add_target(path_str: str, path_obj: Path) -> None:
        if path_str in target_set:
            return
        targets.append(path_obj)
        target_set.add(path_str)

    for path_str, path_obj in current_map.items():
        current_stat = get_file_stat(path_str)
        if not current_stat:
            add_target(path_str, path_obj)
            current_file_states[path_str] = {
                "path": path_str,
                "last_indexed_at": time.time(),
                "deletion_candidate_since": None,
            }
            continue

        prev_state = prev_file_states.get(path_str)
        should_reindex, new_state = should_reindex_file(
            path_str, current_stat, prev_state, diff_mode, fast_fp_bytes, full_hash_algo
        )

        if should_reindex or path_str in retry_set:
            add_target(path_str, path_obj)

        current_file_states[path_str] = new_state

    updated_data: Dict[str, Dict] = {}
    if targets:
        max_workers = max(1, os.cpu_count() or 4)
        log_info(f"インデックス更新開始: {folder} 対象={len(targets)} workers={max_workers}")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_path = {
                pool.submit(extract_text_from_file_with_reason, str(p)): str(p) for p in targets
            }
            for future in future_to_path:
                path_str = future_to_path[future]
                try:
                    texts, reason = future.result()
                    if texts:
                        updated_data[path_str] = {
                            "mtime": os.path.getmtime(path_str),
                            "data": texts,
                            "high_acc": True,
                        }
                        if path_str in failures:
                            failures.pop(path_str, None)
                        if path_str in retry_set:
                            log_notice(f"再試行成功: {path_str}")
                    else:
                        failures[path_str] = reason or "抽出結果が空"
                        log_warn(f"インデックス失敗: {path_str} 理由={failures[path_str]}")
                except Exception:
                    failures[path_str] = "例外"
                    log_warn(f"インデックス失敗: {path_str} 理由=例外")
                    continue
        log_info(f"インデックス更新完了: {folder} 更新={len(updated_data)}")

    valid_cache.update(updated_data)
    failures = {k: v for k, v in failures.items() if k not in valid_cache}
    save_index_to_disk(folder, valid_cache, gen_dir)

    save_file_state(folder, current_file_states, gen_dir)
    save_failures(folder, failures, gen_dir)

    stats = {
        "total_files": len(all_files),
        "indexed_files": len(valid_cache),
        "updated_files": len(updated_data),
        "skipped_files": len(valid_cache) - len(updated_data),
    }
    elapsed = time.time() - start_time
    log_info(f"インデックス構築完了: {folder} 件数={stats['indexed_files']} 時間={elapsed:.1f}s")

    if failures:
        debug_mode = os.getenv("SEARCH_DEBUG", "").strip().lower() in {"1", "true", "yes"}
        if debug_mode:
            for i, (path, reason) in enumerate(list(failures.items())[:5], 1):
                log_warn(f"  [{i}] {path}: {reason}")
            if len(failures) > 5:
                log_warn(f"  ... 他 {len(failures) - 5} 件")

    return valid_cache, stats, failures
