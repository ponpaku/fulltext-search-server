import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .utils import log_warn

from dotenv import load_dotenv, find_dotenv

SYSTEM_VERSION = "1.2.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
INDEX_DIR = BASE_DIR / "indexes"
INDEX_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"}
INDEX_VERSION = "v3"
SEARCH_CACHE_VERSION = "v2"
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
FIXED_CACHE_INDEX = CACHE_DIR / "fixed_cache_index.json"
QUERY_STATS_PATH = CACHE_DIR / "query_stats.json"

# --- 世代ディレクトリ定数 ---
BUILD_DIR = INDEX_DIR / ".build"
BUILD_DIR.mkdir(exist_ok=True)
CURRENT_POINTER_FILE = INDEX_DIR / "current.txt"

# --- 段階的ハッシング定数 ---
FILE_STATE_FILENAME = "file_state.jsonl"

# --- 検索表示定数 ---
SNIPPET_PREFIX_CHARS = 40
SNIPPET_TOTAL_LENGTH = 160
DETAIL_CONTEXT_PREFIX = 500
DETAIL_WINDOW_SIZE = 2000

# --- 検索パフォーマンス定数 ---
SEARCH_ENTRIES_CHUNK_THRESHOLD = 2000

# --- キャッシュデフォルト定数 ---
DEFAULT_CACHE_MAX_ENTRIES = 200
DEFAULT_CACHE_MAX_MB = 200
DEFAULT_CACHE_MAX_RESULT_KB = 2000
DEFAULT_HEARTBEAT_TTL_SEC = 90

STATIC_DIR = BASE_DIR / "static"
CONFIG_EXAMPLE_PATH = BASE_DIR / "config.example.json"


@dataclass
class AppConfig:
    project_root: Path
    index_dir: Path
    cache_dir: Path
    static_dir: Path
    configured_folders: List[Dict[str, str]]
    host_aliases: Dict[str, str]


def read_env_raw_value(var_name: str) -> str | None:
    env_path = find_dotenv()
    if not env_path:
        return None
    try:
        with open(env_path, "r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.lower().startswith("export "):
                    stripped = stripped[7:].lstrip()
                if "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() != var_name:
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                return value
    except OSError:
        return None
    return None


def load_env() -> None:
    had_search_folders = "SEARCH_FOLDERS" in os.environ
    raw_search_folders = read_env_raw_value("SEARCH_FOLDERS")
    load_dotenv()
    if not had_search_folders and raw_search_folders is not None:
        os.environ["SEARCH_FOLDERS"] = raw_search_folders


def _resolve_config_path() -> Path:
    raw = os.getenv("CONFIG_PATH", "").strip() or "config.json"
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _stringify_config_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _set_env_if_missing(name: str, value: object | None) -> None:
    if value is None or name in os.environ:
        return
    os.environ[name] = _stringify_config_value(value)


def _build_search_folders_value(folders: list) -> str:
    entries: list[str] = []
    for entry in folders:
        label = ""
        path = ""
        if isinstance(entry, dict):
            label = str(entry.get("label") or "")
            path = str(entry.get("path") or "")
        elif isinstance(entry, str):
            path = entry
        if not path:
            continue
        entries.append(f"{label}={path}" if label else path)
    return ";".join(entries)


def _build_aliases_value(aliases: dict) -> str:
    entries: list[str] = []
    for key, value in aliases.items():
        if key is None or value is None:
            continue
        key_str = str(key).strip()
        value_str = str(value).strip()
        if key_str and value_str:
            entries.append(f"{key_str}={value_str}")
    return ";".join(entries)


def _apply_config_env(config: dict) -> None:
    if not isinstance(config, dict):
        return

    search = config.get("search")
    if isinstance(search, dict):
        folders = search.get("folders")
        if isinstance(folders, list):
            folders_value = _build_search_folders_value(folders)
            if folders_value:
                _set_env_if_missing("SEARCH_FOLDERS", folders_value)

        aliases = search.get("folder_aliases")
        if isinstance(aliases, dict):
            aliases_value = _build_aliases_value(aliases)
            if aliases_value:
                _set_env_if_missing("SEARCH_FOLDER_ALIASES", aliases_value)

        _set_env_if_missing("SEARCH_EXECUTION_MODE", search.get("execution_mode"))
        _set_env_if_missing("SEARCH_PROCESS_SHARED", search.get("process_shared"))
        _set_env_if_missing("SEARCH_CPU_BUDGET", search.get("cpu_budget"))
        _set_env_if_missing("SEARCH_CONCURRENCY", search.get("concurrency"))
        _set_env_if_missing("SEARCH_WORKERS", search.get("workers"))
        if search.get("query_normalize") is not None:
            _set_env_if_missing("QUERY_NORMALIZE", search.get("query_normalize"))

    front = config.get("front")
    if isinstance(front, dict):
        front_map = {
            "results_batch_size": "FRONT_RESULTS_BATCH_SIZE",
            "results_scroll_threshold_px": "FRONT_RESULTS_SCROLL_THRESHOLD_PX",
            "history_max_items": "FRONT_HISTORY_MAX_ITEMS",
            "range_max": "FRONT_RANGE_MAX",
            "range_default": "FRONT_RANGE_DEFAULT",
            "space_mode_default": "FRONT_SPACE_MODE_DEFAULT",
            "normalize_mode_default": "FRONT_NORMALIZE_MODE_DEFAULT",
            "heartbeat_interval_ms": "FRONT_HEARTBEAT_INTERVAL_MS",
            "heartbeat_jitter_ms": "FRONT_HEARTBEAT_JITTER_MS",
            "heartbeat_min_gap_ms": "FRONT_HEARTBEAT_MIN_GAP_MS",
            "heartbeat_interaction_gap_ms": "FRONT_HEARTBEAT_INTERACTION_GAP_MS",
            "heartbeat_idle_threshold_ms": "FRONT_HEARTBEAT_IDLE_THRESHOLD_MS",
            "heartbeat_fail_threshold": "FRONT_HEARTBEAT_FAIL_THRESHOLD",
            "heartbeat_stale_multiplier": "FRONT_HEARTBEAT_STALE_MULTIPLIER",
            "health_check_interval_ms": "FRONT_HEALTH_CHECK_INTERVAL_MS",
            "health_check_jitter_ms": "FRONT_HEALTH_CHECK_JITTER_MS",
        }
        for key, env_name in front_map.items():
            if key in front:
                _set_env_if_missing(env_name, front.get(key))

    query_stats = config.get("query_stats")
    if isinstance(query_stats, dict):
        _set_env_if_missing("QUERY_STATS_TTL_DAYS", query_stats.get("ttl_days"))
        _set_env_if_missing("QUERY_STATS_FLUSH_SEC", query_stats.get("flush_sec"))

    cache = config.get("cache")
    if isinstance(cache, dict):
        cache_map = {
            "fixed_min_count": "CACHE_FIXED_MIN_COUNT",
            "fixed_min_time_ms": "CACHE_FIXED_MIN_TIME_MS",
            "fixed_min_hits": "CACHE_FIXED_MIN_HITS",
            "fixed_min_kb": "CACHE_FIXED_MIN_KB",
            "fixed_ttl_days": "CACHE_FIXED_TTL_DAYS",
            "fixed_max_entries": "CACHE_FIXED_MAX_ENTRIES",
            "fixed_trigger_cooldown_sec": "CACHE_FIXED_TRIGGER_COOLDOWN_SEC",
            "mem_max_mb": "CACHE_MEM_MAX_MB",
            "mem_max_entries": "CACHE_MEM_MAX_ENTRIES",
            "mem_max_result_kb": "CACHE_MEM_MAX_RESULT_KB",
            "compress_min_kb": "CACHE_COMPRESS_MIN_KB",
        }
        for key, env_name in cache_map.items():
            if key in cache:
                _set_env_if_missing(env_name, cache.get(key))

    rebuild = config.get("rebuild")
    if isinstance(rebuild, dict):
        _set_env_if_missing("REBUILD_SCHEDULE", rebuild.get("schedule"))
        if "allow_shrink" in rebuild:
            _set_env_if_missing("REBUILD_ALLOW_SHRINK", rebuild.get("allow_shrink"))

    index = config.get("index")
    if isinstance(index, dict):
        index_map = {
            "keep_generations": "INDEX_KEEP_GENERATIONS",
            "keep_days": "INDEX_KEEP_DAYS",
            "max_bytes": "INDEX_MAX_BYTES",
            "cleanup_grace_sec": "INDEX_CLEANUP_GRACE_SEC",
            "store_normalized": "INDEX_STORE_NORMALIZED",
        }
        for key, env_name in index_map.items():
            if key in index:
                _set_env_if_missing(env_name, index.get(key))

    diff = config.get("diff")
    if isinstance(diff, dict):
        _set_env_if_missing("DIFF_MODE", diff.get("mode"))
        _set_env_if_missing("FAST_FP_BYTES", diff.get("fast_fp_bytes"))
        _set_env_if_missing("FULL_HASH_ALGO", diff.get("full_hash_algo"))
        full_hash_paths = diff.get("full_hash_paths")
        if isinstance(full_hash_paths, list) and full_hash_paths:
            _set_env_if_missing("FULL_HASH_PATHS", ",".join(str(p) for p in full_hash_paths if p))
        full_hash_exts = diff.get("full_hash_exts")
        if isinstance(full_hash_exts, list) and full_hash_exts:
            _set_env_if_missing("FULL_HASH_EXTS", ",".join(str(e) for e in full_hash_exts if e))

    if "query_normalize" in config:
        _set_env_if_missing("QUERY_NORMALIZE", config.get("query_normalize"))
    if "index_store_normalized" in config:
        _set_env_if_missing("INDEX_STORE_NORMALIZED", config.get("index_store_normalized"))


def _load_json_config() -> dict:
    config_path = _resolve_config_path()
    if not config_path.exists():
        if CONFIG_EXAMPLE_PATH.exists():
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(CONFIG_EXAMPLE_PATH, config_path)
                log_warn(
                    "config.json が見つからないため config.example.json をコピーしました。"
                    f" 設定後に再起動してください: {config_path}"
                )
            except OSError as exc:
                log_warn(f"config.json の作成に失敗しました: {exc}")
        raise RuntimeError(f"config.json が見つかりません: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"config.json の解析に失敗しました: {exc}") from exc


def parse_host_aliases() -> Dict[str, str]:
    raw = os.getenv("SEARCH_FOLDER_ALIASES", "")
    normalized = raw.replace("\n", ";").replace("|", ";").replace(",", ";")
    aliases: Dict[str, str] = {}
    for part in normalized.split(";"):
        entry = part.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            aliases[key] = value
    return aliases


def display_path_for_path(path: str, aliases: Dict[str, str]) -> str:
    if not aliases:
        return path
    match = re.match(r"^(\\\\|//)([^\\\\/]+)([\\\\/].*)?$", path)
    if not match:
        for host, alias in aliases.items():
            pattern = re.compile(rf"(\\\\|//){re.escape(host)}(?=[\\\\/])", re.IGNORECASE)
            if pattern.search(path):
                return pattern.sub(lambda m: f"{m.group(1)}{alias}", path)
        return path
    prefix, host, rest = match.groups()
    host_key = host.lower()
    alias = aliases.get(host_key)
    if not alias:
        return path
    return f"{prefix}{alias}{rest or ''}"


def parse_configured_folders() -> List[Dict[str, str]]:
    raw = os.getenv("SEARCH_FOLDERS", "")
    normalized = raw.replace("\n", ";").replace("|", ";").replace(",", ";")
    folders: List[Dict[str, str]] = []
    for part in normalized.split(";"):
        entry = part.strip()
        if not entry:
            continue
        if "#" in entry:
            entry = entry.split("#", 1)[0].strip()
        if not entry:
            continue
        label = None
        if "=" in entry:
            label, entry = entry.split("=", 1)
            label = label.strip()
            entry = entry.strip()
        p = entry.strip().strip('"').strip("'")
        if not p:
            continue
        # smb://host/share または smb:\host\share を UNC/posix 共有パスに変換
        if p.lower().startswith("smb:"):
            without_scheme = p[4:].lstrip("/\\")
            if not without_scheme:
                continue
            if os.name == "nt":
                unc_path = without_scheme.replace("/", "\\")
                p = f"\\\\{unc_path}"
            else:
                posix_path = without_scheme.replace("\\", "/")
                p = f"//{posix_path}"
        folders.append(
            {
                "path": os.path.abspath(os.path.expanduser(p)),
                "label": label or "",
            }
        )
    return folders


def load_config() -> AppConfig:
    load_env()
    config_data = _load_json_config()
    _apply_config_env(config_data)
    return AppConfig(
        project_root=PROJECT_ROOT,
        index_dir=INDEX_DIR,
        cache_dir=CACHE_DIR,
        static_dir=STATIC_DIR,
        configured_folders=parse_configured_folders(),
        host_aliases=parse_host_aliases(),
    )
