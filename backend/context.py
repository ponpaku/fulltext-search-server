import asyncio
import mmap
import threading
from dataclasses import dataclass, field
from typing import Dict, List

from .config import AppConfig

SYSTEM_VERSION = "1.2.0"


@dataclass
class AppState:
    configured_folders: List[Dict[str, str]] = field(default_factory=list)
    host_aliases: Dict[str, str] = field(default_factory=dict)
    folder_states: Dict[str, Dict] = field(default_factory=dict)
    memory_indexes: Dict[str, Dict[str, Dict]] = field(default_factory=dict)
    memory_pages: Dict[str, List[Dict]] = field(default_factory=dict)
    index_failures: Dict[str, Dict[str, str]] = field(default_factory=dict)
    cache_lock: threading.RLock = field(default_factory=threading.RLock)
    memory_cache: "MemoryCache | None" = None
    query_stats: Dict[str, Dict] = field(default_factory=dict)
    fixed_cache_index: Dict[str, Dict] = field(default_factory=dict)
    fixed_cache_dirty: bool = False
    fixed_cache_last_saved: float = 0.0
    query_stats_dirty: bool = False
    query_stats_last_saved: float = 0.0
    fixed_cache_rebuild_in_progress: bool = False
    fixed_cache_last_trigger: float = 0.0
    folder_order: Dict[str, int] = field(default_factory=dict)
    access_urls: List[str] = field(default_factory=list)
    process_shared: bool = False
    worker_shared_entries: Dict[str, List[Dict]] = field(default_factory=dict)
    worker_shared_mm: Dict[str, mmap.mmap] = field(default_factory=dict)
    worker_shared_files: Dict[str, object] = field(default_factory=dict)
    search_semaphore: asyncio.Semaphore | None = None
    rw_lock: "AsyncRWLock | None" = None
    search_worker_count: int = 1
    search_concurrency: int = 1
    search_executor: "ProcessPoolExecutor | None" = None
    search_execution_mode: str = "thread"
    normalize_mode_warning_emitted: bool = False
    active_client_heartbeats: Dict[str, float] = field(default_factory=dict)
    active_client_lock: threading.Lock = field(default_factory=threading.Lock)
    # Warmup state
    last_search_ts: float = 0.0
    last_warmup_ts: float = 0.0


@dataclass
class AppContext:
    config: AppConfig
    state: AppState
