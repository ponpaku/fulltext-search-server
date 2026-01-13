"""System Version: 1.1.11
File Version: 1.0.0"""
from __future__ import annotations

try:
    from colorama import Fore, Style, init as colorama_init
except Exception:
    Fore = None
    Style = None
    colorama_init = None


_COLOR_ENABLED = False


def _ensure_color() -> None:
    global _COLOR_ENABLED
    if _COLOR_ENABLED or colorama_init is None:
        return
    try:
        colorama_init()
        _COLOR_ENABLED = True
    except Exception:
        _COLOR_ENABLED = False


def log_info(message: str) -> None:
    if Fore is None:
        print(message)
        return
    _ensure_color()
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {message}")


def log_warn(message: str) -> None:
    if Fore is None:
        print(message)
        return
    _ensure_color()
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {message}")


def log_notice(message: str) -> None:
    if Fore is None:
        print(message)
        return
    _ensure_color()
    print(f"{Fore.MAGENTA}[NOTICE]{Style.RESET_ALL} {message}")


def log_success(message: str) -> None:
    if Fore is None:
        print(message)
        return
    _ensure_color()
    print(f"{Fore.GREEN}[OK]{Style.RESET_ALL} {message}")


def colorize_url(url: str) -> str:
    if Fore is None:
        return url
    _ensure_color()
    return f"{Fore.GREEN}{url}{Style.RESET_ALL}"
