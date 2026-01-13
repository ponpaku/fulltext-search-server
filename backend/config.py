import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv, find_dotenv

SYSTEM_VERSION = "1.1.11"
# File Version: 1.0.0

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
