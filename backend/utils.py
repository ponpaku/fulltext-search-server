"""Utility functions for logging, environment variables, and network."""
from __future__ import annotations

import hashlib
import os
import re
import socket
import subprocess
import time
from typing import List

SYSTEM_VERSION = "1.2.0"

try:
    from colorama import Fore, Style, init as colorama_init
except Exception:
    Fore = None
    Style = None
    colorama_init = None


def log_info(message: str):
    if colorama_init:
        colorama_init()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def log_warn(message: str):
    if colorama_init:
        colorama_init()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if Fore and Style:
        print(f"{Fore.RED}[{timestamp}] {message}{Style.RESET_ALL}")
    else:
        print(f"[{timestamp}] {message}")


def log_notice(message: str):
    """Yellow notification log for moderate importance warnings."""
    if colorama_init:
        colorama_init()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if Fore and Style:
        print(f"{Fore.YELLOW}[{timestamp}] {message}{Style.RESET_ALL}")
    else:
        print(f"[{timestamp}] {message}")


def log_success(message: str):
    if colorama_init:
        colorama_init()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if Fore and Style:
        print(f"{Fore.GREEN}[{timestamp}] {message}{Style.RESET_ALL}")
    else:
        print(f"[{timestamp}] {message}")


def colorize_url(url: str) -> str:
    if Fore and Style:
        return f"{Fore.CYAN}{url}{Style.RESET_ALL}"
    return url


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if raw.isdigit():
        return int(raw)
    return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw)
    except Exception:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    return default


def get_ipv4_addresses() -> List[str]:
    ips: set[str] = set()
    try:
        hostname = socket.gethostname()
        for item in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = item[4][0]
            if ip and not ip.startswith("127."):
                ips.add(ip)
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if ip and not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass

    if os.name == "nt":
        try:
            output = subprocess.check_output(["ipconfig"], text=True, errors="ignore")
            for line in output.splitlines():
                line = line.strip()
                if "IPv4 Address" in line or "IPv4 アドレス" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        ip = parts[-1].strip()
                        if ip and not ip.startswith("127."):
                            ips.add(ip)
        except Exception:
            pass
    else:
        try:
            output = subprocess.check_output(["/sbin/ip", "-4", "addr"], text=True, errors="ignore")
            for match in re.finditer(r"\binet\s+(\d+\.\d+\.\d+\.\d+)", output):
                ip = match.group(1)
                if ip and not ip.startswith("127."):
                    ips.add(ip)
        except Exception:
            pass
    return sorted(ips)


def folder_id_from_path(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:10]


def file_id_from_path(path: str) -> str:
    # 12 hex chars = 48 bits, collision probability ~1e-7 at 10k files
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
