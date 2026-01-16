import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

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
CONFIG_EXAMPLE_PATH = PROJECT_ROOT / "config.example.json"


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
    raw_path = os.getenv("CONFIG_PATH", "").strip() or "config.json"
    config_path = Path(raw_path).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return config_path


def _ensure_config_file(config_path: Path) -> None:
    if config_path.exists():
        return
    if CONFIG_EXAMPLE_PATH.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CONFIG_EXAMPLE_PATH, config_path)
    raise RuntimeError(
        "config.json が見つかりません。config.example.json をコピーしたので、"
        "内容を設定してから再起動してください。"
    )


def _load_json_config() -> Dict[str, Any]:
    config_path = _resolve_config_path()
    _ensure_config_file(config_path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"config.json の解析に失敗しました: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("config.json はオブジェクト形式で定義してください。")
    return data


def _set_env_if_absent(key: str, value: str | None) -> None:
    if value is None or key in os.environ:
        return
    os.environ[key] = value


def _to_env_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _folders_to_env(folders: Any) -> str | None:
    if not isinstance(folders, list):
        return None
    parts: List[str] = []
    for entry in folders:
        if isinstance(entry, str):
            path = entry.strip()
            if path:
                parts.append(path)
            continue
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path", "")).strip()
        if not path:
            continue
        label = str(entry.get("label", "")).strip()
        if label:
            parts.append(f"{label}={path}")
        else:
            parts.append(path)
    return ",".join(parts) if parts else None


def _aliases_to_env(aliases: Any) -> str | None:
    if not isinstance(aliases, dict):
        return None
    parts: List[str] = []
    for key, value in aliases.items():
        key_text = str(key).strip()
        value_text = str(value).strip()
        if key_text and value_text:
            parts.append(f"{key_text}={value_text}")
    return ",".join(parts) if parts else None


def _list_to_env(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(items) if items else None
    text = str(value).strip()
    return text or None


def apply_config_defaults(config: Dict[str, Any]) -> None:
    search = config.get("search", {}) if isinstance(config.get("search"), dict) else {}
    _set_env_if_absent("SEARCH_FOLDERS", _folders_to_env(search.get("folders")))
    _set_env_if_absent("SEARCH_FOLDER_ALIASES", _aliases_to_env(search.get("folder_aliases")))
    _set_env_if_absent("SEARCH_EXECUTION_MODE", _to_env_value(search.get("execution_mode")))
    _set_env_if_absent("SEARCH_PROCESS_SHARED", _to_env_value(search.get("process_shared")))
    _set_env_if_absent("SEARCH_CPU_BUDGET", _to_env_value(search.get("cpu_budget")))
    _set_env_if_absent("SEARCH_CONCURRENCY", _to_env_value(search.get("concurrency")))
    _set_env_if_absent("SEARCH_WORKERS", _to_env_value(search.get("workers")))

    front = config.get("front", {}) if isinstance(config.get("front"), dict) else {}
    _set_env_if_absent("FRONT_RESULTS_BATCH_SIZE", _to_env_value(front.get("results_batch_size")))
    _set_env_if_absent("FRONT_RANGE_DEFAULT", _to_env_value(front.get("range_default")))
    _set_env_if_absent("FRONT_SPACE_MODE_DEFAULT", _to_env_value(front.get("space_mode_default")))
    _set_env_if_absent("FRONT_NORMALIZE_MODE_DEFAULT", _to_env_value(front.get("normalize_mode_default")))
    _set_env_if_absent("FRONT_RESULTS_SCROLL_THRESHOLD_PX", _to_env_value(front.get("results_scroll_threshold_px")))
    _set_env_if_absent("FRONT_HISTORY_MAX_ITEMS", _to_env_value(front.get("history_max_items")))
    _set_env_if_absent("FRONT_RANGE_MAX", _to_env_value(front.get("range_max")))
    _set_env_if_absent("FRONT_HEARTBEAT_INTERVAL_MS", _to_env_value(front.get("heartbeat_interval_ms")))
    _set_env_if_absent("FRONT_HEARTBEAT_JITTER_MS", _to_env_value(front.get("heartbeat_jitter_ms")))
    _set_env_if_absent("FRONT_HEARTBEAT_MIN_GAP_MS", _to_env_value(front.get("heartbeat_min_gap_ms")))
    _set_env_if_absent(
        "FRONT_HEARTBEAT_INTERACTION_GAP_MS", _to_env_value(front.get("heartbeat_interaction_gap_ms"))
    )
    _set_env_if_absent(
        "FRONT_HEARTBEAT_IDLE_THRESHOLD_MS", _to_env_value(front.get("heartbeat_idle_threshold_ms"))
    )
    _set_env_if_absent("FRONT_HEARTBEAT_FAIL_THRESHOLD", _to_env_value(front.get("heartbeat_fail_threshold")))
    _set_env_if_absent(
        "FRONT_HEARTBEAT_STALE_MULTIPLIER", _to_env_value(front.get("heartbeat_stale_multiplier"))
    )
    _set_env_if_absent("FRONT_HEALTH_CHECK_INTERVAL_MS", _to_env_value(front.get("health_check_interval_ms")))
    _set_env_if_absent("FRONT_HEALTH_CHECK_JITTER_MS", _to_env_value(front.get("health_check_jitter_ms")))

    query = config.get("query", {}) if isinstance(config.get("query"), dict) else {}
    _set_env_if_absent("QUERY_NORMALIZE", _to_env_value(query.get("normalize")))

    query_stats = config.get("query_stats", {}) if isinstance(config.get("query_stats"), dict) else {}
    _set_env_if_absent("QUERY_STATS_TTL_DAYS", _to_env_value(query_stats.get("ttl_days")))
    _set_env_if_absent("QUERY_STATS_FLUSH_SEC", _to_env_value(query_stats.get("flush_sec")))

    cache = config.get("cache", {}) if isinstance(config.get("cache"), dict) else {}
    _set_env_if_absent("CACHE_FIXED_MIN_COUNT", _to_env_value(cache.get("fixed_min_count")))
    _set_env_if_absent("CACHE_FIXED_MIN_TIME_MS", _to_env_value(cache.get("fixed_min_time_ms")))
    _set_env_if_absent("CACHE_FIXED_MIN_HITS", _to_env_value(cache.get("fixed_min_hits")))
    _set_env_if_absent("CACHE_FIXED_MIN_KB", _to_env_value(cache.get("fixed_min_kb")))
    _set_env_if_absent("CACHE_FIXED_TTL_DAYS", _to_env_value(cache.get("fixed_ttl_days")))
    _set_env_if_absent("CACHE_FIXED_MAX_ENTRIES", _to_env_value(cache.get("fixed_max_entries")))
    _set_env_if_absent(
        "CACHE_FIXED_TRIGGER_COOLDOWN_SEC", _to_env_value(cache.get("fixed_trigger_cooldown_sec"))
    )
    _set_env_if_absent("CACHE_MEM_MAX_MB", _to_env_value(cache.get("mem_max_mb")))
    _set_env_if_absent("CACHE_MEM_MAX_ENTRIES", _to_env_value(cache.get("mem_max_entries")))
    _set_env_if_absent("CACHE_MEM_MAX_RESULT_KB", _to_env_value(cache.get("mem_max_result_kb")))
    _set_env_if_absent("CACHE_COMPRESS_MIN_KB", _to_env_value(cache.get("compress_min_kb")))

    rebuild = config.get("rebuild", {}) if isinstance(config.get("rebuild"), dict) else {}
    _set_env_if_absent("REBUILD_SCHEDULE", _to_env_value(rebuild.get("schedule")))
    _set_env_if_absent("REBUILD_ALLOW_SHRINK", _to_env_value(rebuild.get("allow_shrink")))

    index = config.get("index", {}) if isinstance(config.get("index"), dict) else {}
    _set_env_if_absent("INDEX_KEEP_GENERATIONS", _to_env_value(index.get("keep_generations")))
    _set_env_if_absent("INDEX_KEEP_DAYS", _to_env_value(index.get("keep_days")))
    _set_env_if_absent("INDEX_MAX_BYTES", _to_env_value(index.get("max_bytes")))
    _set_env_if_absent("INDEX_CLEANUP_GRACE_SEC", _to_env_value(index.get("cleanup_grace_sec")))
    _set_env_if_absent("INDEX_STORE_NORMALIZED", _to_env_value(index.get("store_normalized")))

    diff = config.get("diff", {}) if isinstance(config.get("diff"), dict) else {}
    _set_env_if_absent("DIFF_MODE", _to_env_value(diff.get("mode")))
    _set_env_if_absent("FAST_FP_BYTES", _to_env_value(diff.get("fast_fp_bytes")))
    _set_env_if_absent("FULL_HASH_ALGO", _to_env_value(diff.get("full_hash_algo")))
    _set_env_if_absent("FULL_HASH_PATHS", _list_to_env(diff.get("full_hash_paths")))
    _set_env_if_absent("FULL_HASH_EXTS", _list_to_env(diff.get("full_hash_exts")))


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
    apply_config_defaults(config_data)
    return AppConfig(
        project_root=PROJECT_ROOT,
        index_dir=INDEX_DIR,
        cache_dir=CACHE_DIR,
        static_dir=STATIC_DIR,
        configured_folders=parse_configured_folders(),
        host_aliases=parse_host_aliases(),
    )
