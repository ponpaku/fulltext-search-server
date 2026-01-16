import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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
    apply_config_json()


def _set_env_if_missing(key: str, value: Any) -> None:
    if value is None:
        return
    current = os.environ.get(key)
    if current is not None and current != "":
        return
    if isinstance(value, bool):
        os.environ[key] = "1" if value else "0"
        return
    if isinstance(value, (int, float)):
        os.environ[key] = str(value)
        return
    text = str(value).strip()
    if text:
        os.environ[key] = text


def _serialize_folders(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    parts: List[str] = []
    for item in value:
        if isinstance(item, str):
            entry = item.strip()
            if entry:
                parts.append(entry)
            continue
        if isinstance(item, dict):
            path = str(item.get("path", "")).strip()
            if not path:
                continue
            label = str(item.get("label", "")).strip()
            parts.append(f"{label}={path}" if label else path)
    return ";".join(parts)


def _serialize_kv_map(value: Any) -> str:
    if isinstance(value, dict):
        entries = []
        for key, val in value.items():
            key_str = str(key).strip()
            val_str = str(val).strip()
            if key_str and val_str:
                entries.append(f"{key_str}={val_str}")
        return ";".join(entries)
    if isinstance(value, str):
        return value.strip()
    return ""


def _serialize_list(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(items)
    return ""


def load_config_json() -> Dict[str, Any]:
    config_path = os.getenv("CONFIG_PATH", "").strip()
    path = Path(config_path) if config_path else BASE_DIR / "config.json"
    path = path.expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if not path.exists():
        example_path = BASE_DIR / "config.example.json"
        if example_path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                shutil.copy(example_path, path)
            raise RuntimeError(
                f"{path} が見つからないため config.example.json をコピーしました。"
                "内容を編集して再起動してください。"
            )
        raise RuntimeError(
            f"{path} が見つかりません。config.example.json を作成して設定してください。"
        )
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} の読み込みに失敗しました: {exc}") from exc


def apply_config_json() -> None:
    config = load_config_json()
    if not config:
        return

    search = config.get("search", {})
    _set_env_if_missing("SEARCH_FOLDERS", _serialize_folders(search.get("folders")))
    _set_env_if_missing(
        "SEARCH_FOLDER_ALIASES", _serialize_kv_map(search.get("folder_aliases"))
    )
    _set_env_if_missing("SEARCH_EXECUTION_MODE", search.get("execution_mode"))
    _set_env_if_missing("SEARCH_PROCESS_SHARED", search.get("process_shared"))
    _set_env_if_missing("SEARCH_CPU_BUDGET", search.get("cpu_budget"))
    _set_env_if_missing("SEARCH_CONCURRENCY", search.get("concurrency"))
    _set_env_if_missing("SEARCH_WORKERS", search.get("workers"))

    front = config.get("front", {})
    _set_env_if_missing("FRONT_RESULTS_BATCH_SIZE", front.get("results_batch_size"))
    _set_env_if_missing(
        "FRONT_RESULTS_SCROLL_THRESHOLD_PX",
        front.get("results_scroll_threshold_px"),
    )
    _set_env_if_missing("FRONT_HISTORY_MAX_ITEMS", front.get("history_max_items"))
    _set_env_if_missing("FRONT_RANGE_MAX", front.get("range_max"))
    _set_env_if_missing("FRONT_RANGE_DEFAULT", front.get("range_default"))
    _set_env_if_missing("FRONT_SPACE_MODE_DEFAULT", front.get("space_mode_default"))
    _set_env_if_missing(
        "FRONT_NORMALIZE_MODE_DEFAULT", front.get("normalize_mode_default")
    )
    _set_env_if_missing(
        "FRONT_HEARTBEAT_INTERVAL_MS", front.get("heartbeat_interval_ms")
    )
    _set_env_if_missing(
        "FRONT_HEARTBEAT_JITTER_MS", front.get("heartbeat_jitter_ms")
    )
    _set_env_if_missing(
        "FRONT_HEARTBEAT_MIN_GAP_MS", front.get("heartbeat_min_gap_ms")
    )
    _set_env_if_missing(
        "FRONT_HEARTBEAT_INTERACTION_GAP_MS",
        front.get("heartbeat_interaction_gap_ms"),
    )
    _set_env_if_missing(
        "FRONT_HEARTBEAT_IDLE_THRESHOLD_MS",
        front.get("heartbeat_idle_threshold_ms"),
    )
    _set_env_if_missing(
        "FRONT_HEARTBEAT_FAIL_THRESHOLD", front.get("heartbeat_fail_threshold")
    )
    _set_env_if_missing(
        "FRONT_HEARTBEAT_STALE_MULTIPLIER", front.get("heartbeat_stale_multiplier")
    )
    _set_env_if_missing(
        "FRONT_HEALTH_CHECK_INTERVAL_MS", front.get("health_check_interval_ms")
    )
    _set_env_if_missing(
        "FRONT_HEALTH_CHECK_JITTER_MS", front.get("health_check_jitter_ms")
    )

    query = config.get("query", {})
    normalize_mode = query.get("normalize")
    if normalize_mode is None:
        normalize_mode = config.get("query_normalize")
    _set_env_if_missing("QUERY_NORMALIZE", normalize_mode)

    query_stats = config.get("query_stats", {})
    _set_env_if_missing("QUERY_STATS_TTL_DAYS", query_stats.get("ttl_days"))
    _set_env_if_missing("QUERY_STATS_FLUSH_SEC", query_stats.get("flush_sec"))

    cache = config.get("cache", {})
    _set_env_if_missing("CACHE_FIXED_MIN_COUNT", cache.get("fixed_min_count"))
    _set_env_if_missing("CACHE_FIXED_MIN_TIME_MS", cache.get("fixed_min_time_ms"))
    _set_env_if_missing("CACHE_FIXED_MIN_HITS", cache.get("fixed_min_hits"))
    _set_env_if_missing("CACHE_FIXED_MIN_KB", cache.get("fixed_min_kb"))
    _set_env_if_missing("CACHE_FIXED_TTL_DAYS", cache.get("fixed_ttl_days"))
    _set_env_if_missing("CACHE_FIXED_MAX_ENTRIES", cache.get("fixed_max_entries"))
    _set_env_if_missing(
        "CACHE_FIXED_TRIGGER_COOLDOWN_SEC",
        cache.get("fixed_trigger_cooldown_sec"),
    )
    _set_env_if_missing("CACHE_MEM_MAX_MB", cache.get("mem_max_mb"))
    _set_env_if_missing("CACHE_MEM_MAX_ENTRIES", cache.get("mem_max_entries"))
    _set_env_if_missing("CACHE_MEM_MAX_RESULT_KB", cache.get("mem_max_result_kb"))
    _set_env_if_missing("CACHE_COMPRESS_MIN_KB", cache.get("compress_min_kb"))

    rebuild = config.get("rebuild", {})
    _set_env_if_missing("REBUILD_SCHEDULE", rebuild.get("schedule"))
    _set_env_if_missing("REBUILD_ALLOW_SHRINK", rebuild.get("allow_shrink"))

    index = config.get("index", {})
    _set_env_if_missing("INDEX_KEEP_GENERATIONS", index.get("keep_generations"))
    _set_env_if_missing("INDEX_KEEP_DAYS", index.get("keep_days"))
    _set_env_if_missing("INDEX_MAX_BYTES", index.get("max_bytes"))
    _set_env_if_missing("INDEX_CLEANUP_GRACE_SEC", index.get("cleanup_grace_sec"))
    _set_env_if_missing("INDEX_STORE_NORMALIZED", index.get("store_normalized"))

    diff = config.get("diff", {})
    _set_env_if_missing("DIFF_MODE", diff.get("mode"))
    _set_env_if_missing("FAST_FP_BYTES", diff.get("fast_fp_bytes"))
    _set_env_if_missing("FULL_HASH_ALGO", diff.get("full_hash_algo"))
    _set_env_if_missing("FULL_HASH_PATHS", _serialize_list(diff.get("full_hash_paths")))
    _set_env_if_missing("FULL_HASH_EXTS", _serialize_list(diff.get("full_hash_exts")))


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
    return AppConfig(
        project_root=PROJECT_ROOT,
        index_dir=INDEX_DIR,
        cache_dir=CACHE_DIR,
        static_dir=STATIC_DIR,
        configured_folders=parse_configured_folders(),
        host_aliases=parse_host_aliases(),
    )
