import os
from typing import Dict, List

from dotenv import load_dotenv, find_dotenv

SYSTEM_VERSION = "1.1.11"
# File Version: 1.0.0


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
        # smb://host/share または smb:\\host\\share を UNC/posix 共有パスに変換
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
