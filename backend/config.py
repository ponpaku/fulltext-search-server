import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
CONFIG_DEFAULT_FILENAME = "config.json"
CONFIG_EXAMPLE_FILENAME = "config.example.json"


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


def resolve_config_path() -> Path:
    raw_path = os.getenv("CONFIG_PATH", "").strip()
    candidate = raw_path or CONFIG_DEFAULT_FILENAME
    path = Path(candidate)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json_config() -> Dict:
    config_path = resolve_config_path()
    if not config_path.exists():
        example_path = PROJECT_ROOT / CONFIG_EXAMPLE_FILENAME
        if example_path.exists():
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(example_path, config_path)
            except OSError:
                pass
        raise RuntimeError(
            "config.json が見つかりません。"
            f" {config_path} を作成しましたので内容を編集して再起動してください。"
        )
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"config.json の読み込みに失敗しました: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"config.json の読み込みに失敗しました: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("config.json はオブジェクト形式で指定してください。")
    return data


def set_env_if_missing(name: str, value: object | None) -> None:
    if value is None:
        return
    if os.getenv(name, "").strip():
        return
    if isinstance(value, bool):
        os.environ[name] = "1" if value else "0"
        return
    os.environ[name] = str(value)


def encode_search_folders(items: object) -> str | None:
    if not items:
        return None
    if not isinstance(items, list):
        return None
    parts: List[str] = []
    for item in items:
        if isinstance(item, dict):
            path = str(item.get("path", "")).strip()
            if not path:
                continue
            label = str(item.get("label", "")).strip()
            parts.append(f"{label}={path}" if label else path)
        elif isinstance(item, str):
            candidate = item.strip()
            if candidate:
                parts.append(candidate)
    return ";".join(parts) if parts else None


def encode_aliases(aliases: object) -> str | None:
    if not aliases or not isinstance(aliases, dict):
        return None
    parts: List[str] = []
    for key, value in aliases.items():
        key_str = str(key).strip()
        value_str = str(value).strip()
        if key_str and value_str:
            parts.append(f"{key_str}={value_str}")
    return ";".join(parts) if parts else None


def encode_csv_list(values: object) -> str | None:
    if not values:
        return None
    if isinstance(values, list):
        parts = [str(item).strip() for item in values if str(item).strip()]
        return ",".join(parts) if parts else None
    return str(values).strip() if str(values).strip() else None


def apply_config_to_env(config: Dict) -> None:
    search = config.get("search", {}) if isinstance(config, dict) else {}
    set_env_if_missing("SEARCH_FOLDERS", encode_search_folders(search.get("folders")))
    set_env_if_missing("SEARCH_FOLDER_ALIASES", encode_aliases(search.get("folder_aliases")))
    set_env_if_missing("SEARCH_EXECUTION_MODE", search.get("execution_mode"))
    set_env_if_missing("SEARCH_PROCESS_SHARED", search.get("process_shared"))
    set_env_if_missing("SEARCH_CPU_BUDGET", search.get("cpu_budget"))
    set_env_if_missing("SEARCH_CONCURRENCY", search.get("concurrency"))
    set_env_if_missing("SEARCH_WORKERS", search.get("workers"))

    front = config.get("front", {}) if isinstance(config, dict) else {}
    set_env_if_missing("FRONT_RESULTS_BATCH_SIZE", front.get("results_batch_size"))
    set_env_if_missing("FRONT_RESULTS_SCROLL_THRESHOLD_PX", front.get("scroll_threshold_px"))
    set_env_if_missing("FRONT_HISTORY_MAX_ITEMS", front.get("history_max_items"))
    set_env_if_missing("FRONT_RANGE_MAX", front.get("range_max"))
    set_env_if_missing("FRONT_RANGE_DEFAULT", front.get("range_default"))
    set_env_if_missing("FRONT_SPACE_MODE_DEFAULT", front.get("space_mode_default"))
    set_env_if_missing("FRONT_NORMALIZE_MODE_DEFAULT", front.get("normalize_mode_default"))

    query = config.get("query", {}) if isinstance(config, dict) else {}
    set_env_if_missing("QUERY_NORMALIZE", query.get("normalize"))

    query_stats = config.get("query_stats", {}) if isinstance(config, dict) else {}
    set_env_if_missing("QUERY_STATS_TTL_DAYS", query_stats.get("ttl_days"))
    set_env_if_missing("QUERY_STATS_FLUSH_SEC", query_stats.get("flush_sec"))

    cache = config.get("cache", {}) if isinstance(config, dict) else {}
    set_env_if_missing("CACHE_FIXED_MIN_COUNT", cache.get("fixed_min_count"))
    set_env_if_missing("CACHE_FIXED_MIN_TIME_MS", cache.get("fixed_min_time_ms"))
    set_env_if_missing("CACHE_FIXED_MIN_HITS", cache.get("fixed_min_hits"))
    set_env_if_missing("CACHE_FIXED_MIN_KB", cache.get("fixed_min_kb"))
    set_env_if_missing("CACHE_FIXED_TTL_DAYS", cache.get("fixed_ttl_days"))
    set_env_if_missing("CACHE_FIXED_MAX_ENTRIES", cache.get("fixed_max_entries"))
    set_env_if_missing("CACHE_FIXED_TRIGGER_COOLDOWN_SEC", cache.get("fixed_trigger_cooldown_sec"))
    set_env_if_missing("CACHE_MEM_MAX_MB", cache.get("mem_max_mb"))
    set_env_if_missing("CACHE_MEM_MAX_ENTRIES", cache.get("mem_max_entries"))
    set_env_if_missing("CACHE_MEM_MAX_RESULT_KB", cache.get("mem_max_result_kb"))
    set_env_if_missing("CACHE_COMPRESS_MIN_KB", cache.get("compress_min_kb"))

    rebuild = config.get("rebuild", {}) if isinstance(config, dict) else {}
    set_env_if_missing("REBUILD_SCHEDULE", rebuild.get("schedule"))
    set_env_if_missing("REBUILD_ALLOW_SHRINK", rebuild.get("allow_shrink"))

    index = config.get("index", {}) if isinstance(config, dict) else {}
    set_env_if_missing("INDEX_KEEP_GENERATIONS", index.get("keep_generations"))
    set_env_if_missing("INDEX_KEEP_DAYS", index.get("keep_days"))
    set_env_if_missing("INDEX_MAX_BYTES", index.get("max_bytes"))
    set_env_if_missing("INDEX_CLEANUP_GRACE_SEC", index.get("cleanup_grace_sec"))
    set_env_if_missing("INDEX_STORE_NORMALIZED", index.get("store_normalized"))

    diff = config.get("diff", {}) if isinstance(config, dict) else {}
    set_env_if_missing("DIFF_MODE", diff.get("mode"))
    set_env_if_missing("FAST_FP_BYTES", diff.get("fast_fp_bytes"))
    set_env_if_missing("FULL_HASH_ALGO", diff.get("full_hash_algo"))
    set_env_if_missing("FULL_HASH_PATHS", encode_csv_list(diff.get("full_hash_paths")))
    set_env_if_missing("FULL_HASH_EXTS", encode_csv_list(diff.get("full_hash_exts")))


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
    config_data = load_json_config()
    apply_config_to_env(config_data)
    return AppConfig(
        project_root=PROJECT_ROOT,
        index_dir=INDEX_DIR,
        cache_dir=CACHE_DIR,
        static_dir=STATIC_DIR,
        configured_folders=parse_configured_folders(),
        host_aliases=parse_host_aliases(),
    )
