"""System Version: 1.1.11
File Version: 1.0.0"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AppConfig:
    base_dir: Path
    index_dir: Path
    cache_dir: Path
    static_dir: Path
    build_dir: Path
    current_pointer_file: Path
    system_version: str
    index_version: str
    cache_version: str


@dataclass
class AppState:
    folder_states: Dict[str, Dict]
    memory_indexes: Dict[str, Dict]
    memory_pages: Dict[str, list]
    index_failures: Dict[str, Dict]
    memory_cache: Any
    fixed_cache_index: Dict[str, Dict]
    query_stats: Dict[str, Dict]
    search_semaphore: Any
    rw_lock: Any
    search_executor: Any


@dataclass
class AppContext:
    config: AppConfig
    state: AppState


def build_config(
    base_dir: Path,
    index_dir: Path,
    cache_dir: Path,
    static_dir: Path,
    build_dir: Path,
    current_pointer_file: Path,
    system_version: str,
    index_version: str,
    cache_version: str,
) -> AppConfig:
    return AppConfig(
        base_dir=base_dir,
        index_dir=index_dir,
        cache_dir=cache_dir,
        static_dir=static_dir,
        build_dir=build_dir,
        current_pointer_file=current_pointer_file,
        system_version=system_version,
        index_version=index_version,
        cache_version=cache_version,
    )


def build_state(
    folder_states: Dict[str, Dict],
    memory_indexes: Dict[str, Dict],
    memory_pages: Dict[str, list],
    index_failures: Dict[str, Dict],
    memory_cache: Any,
    fixed_cache_index: Dict[str, Dict],
    query_stats: Dict[str, Dict],
    search_semaphore: Any,
    rw_lock: Any,
    search_executor: Any,
) -> AppState:
    return AppState(
        folder_states=folder_states,
        memory_indexes=memory_indexes,
        memory_pages=memory_pages,
        index_failures=index_failures,
        memory_cache=memory_cache,
        fixed_cache_index=fixed_cache_index,
        query_stats=query_stats,
        search_semaphore=search_semaphore,
        rw_lock=rw_lock,
        search_executor=search_executor,
    )
