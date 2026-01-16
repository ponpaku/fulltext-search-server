import asyncio
import csv
import gzip
import hashlib
import io
import json
import mmap
import os
import pickle
import re
import threading
import time
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

SYSTEM_VERSION = "1.2.0"
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .config import (
    ALLOWED_EXTS,
    BASE_DIR,
    BUILD_DIR,
    CACHE_DIR,
    CURRENT_POINTER_FILE,
    DEFAULT_CACHE_MAX_ENTRIES,
    DEFAULT_CACHE_MAX_MB,
    DEFAULT_CACHE_MAX_RESULT_KB,
    DEFAULT_HEARTBEAT_TTL_SEC,
    DETAIL_CONTEXT_PREFIX,
    DETAIL_WINDOW_SIZE,
    FILE_STATE_FILENAME,
    FIXED_CACHE_INDEX,
    INDEX_DIR,
    INDEX_VERSION,
    QUERY_STATS_PATH,
    SEARCH_CACHE_VERSION,
    SEARCH_ENTRIES_CHUNK_THRESHOLD,
    SNIPPET_PREFIX_CHARS,
    SNIPPET_TOTAL_LENGTH,
    STATIC_DIR,
    display_path_for_path,
    load_config,
)
from .context import AppContext, AppState
from .extractors import extract_text_from_file_with_reason
from .index_ops import (
    build_index_for_folder,
    cleanup_old_generations,
    create_generation_uuid,
    create_manifest,
    get_current_generation_dir,
    get_current_generation_manifest,
    get_current_generation_pointer,
    get_generation_dir,
    index_path_for,
    list_generations,
    load_failures,
    load_index_from_disk,
    save_manifest,
    scan_files,
    set_current_generation_pointer,
)
from .search import (
    build_query_groups,
    init_search_worker,
    init_search_worker_shared,
    perform_search_process,
    search_entries_chunk,
)
from .text_utils import (
    apply_space_mode,
    build_detail_text,
    encode_norm_text,
    normalize_invisible_separators,
    normalize_snippet_text,
    normalize_text,
    normalize_text_minimal,
    normalize_text_nfkc_casefold,
)
from .utils import (
    colorize_url,
    env_bool,
    env_float,
    env_int,
    file_id_from_path,
    folder_id_from_path,
    get_ipv4_addresses,
    is_primary_process,
    log_info,
    log_notice,
    log_success,
    log_warn,
    process_role,
)
from .warmup import (
    cleanup_old_warmup_locks,
    keep_warm_loop as warmup_keep_warm_loop,
    trigger_warmup as warmup_trigger_warmup,
)


def perform_search(
    target_ids: List[str],
    params: "SearchParams",
    norm_keyword_groups: List[List[str]],
    raw_keywords: List[str],
    worker_count: int,
    include_detail: bool,
) -> List[Dict]:
    results: List[Dict] = []

    def search_folder(fid: str) -> List[Dict]:
        folder_name = folder_states[fid]["name"]
        entries = memory_pages.get(fid)
        if entries is None:
            cache = memory_indexes.get(fid, {})
            entries = build_memory_pages(fid, folder_name, cache)
            memory_pages[fid] = entries
        if os.getenv("SEARCH_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
            log_info(
                f"検索対象: {folder_states[fid]['path']} entries={len(entries)}"
            )
        folder_results: List[Dict] = []
        entries_iter = entries
        if len(entries_iter) >= SEARCH_ENTRIES_CHUNK_THRESHOLD:
            chunk_workers = max(1, worker_count)
            chunk_size = max(1, (len(entries_iter) + chunk_workers - 1) // chunk_workers)
            chunks = [
                entries_iter[i : i + chunk_size]
                for i in range(0, len(entries_iter), chunk_size)
            ]
            with ThreadPoolExecutor(max_workers=chunk_workers) as executor:
                futures = [
                    executor.submit(
                        search_entries_chunk,
                        chunk,
                        norm_keyword_groups,
                        raw_keywords,
                        params.mode,
                        params.range_limit,
                        params.space_mode,
                        params.normalize_mode,
                        include_detail,
                    )
                    for chunk in chunks
                ]
                for future in futures:
                    try:
                        res = future.result()
                        if res:
                            folder_results.extend(res)
                    except Exception:
                        continue
        else:
            folder_results = search_entries_chunk(
                entries_iter,
                norm_keyword_groups,
                raw_keywords,
                params.mode,
                params.range_limit,
                params.space_mode,
                params.normalize_mode,
                include_detail,
            )
        if os.getenv("SEARCH_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
            log_info(
                f"検索結果: {folder_states[fid]['path']} hits={len(folder_results)}"
            )
        return folder_results

    for fid in target_ids:
        try:
            res = search_folder(fid)
            if res:
                results.extend(res)
        except Exception:
            continue
    return results


def run_search_direct(
    query: str,
    params: "SearchParams",
    target_ids: List[str],
    include_detail: bool = False,
) -> Tuple[List[Dict], List[str]]:
    raw_keywords, norm_keyword_groups = build_query_groups(
        query,
        params.space_mode,
        params.normalize_mode,
    )
    worker_count = per_request_workers()
    if search_execution_mode == "process" and search_executor:
        future = search_executor.submit(
            perform_search_process,
            target_ids,
            params,
            norm_keyword_groups,
            raw_keywords,
            worker_count,
            include_detail,
        )
        results = future.result()
    else:
        results = perform_search(
            target_ids,
            params,
            norm_keyword_groups,
            raw_keywords,
            worker_count,
            include_detail,
        )
    return results, raw_keywords


# --- インデックス関連 ---

# --- 世代管理ユーティリティ ---
def build_memory_pages(folder_id: str, folder_name: str, cache: Dict[str, Dict]) -> List[Dict]:
    store_normalized = env_bool("INDEX_STORE_NORMALIZED", True)
    pages: List[Dict] = []
    for path, meta in cache.items():
        file_name = os.path.basename(path)
        is_pdf = file_name.lower().endswith(".pdf")
        ext = os.path.splitext(file_name)[1].lower()
        is_excel = ext in {".xlsx", ".xls"}
        display_path = display_path_for_path(path, host_aliases)
        file_id = register_file_id(path, folder_id)
        for page_num, raw_text in meta.get("data", {}).items():
            if not raw_text:
                continue
            norm_base = normalize_text(raw_text)
            norm_strict = normalize_text_minimal(raw_text)
            norm_cf_base = (
                normalize_text_nfkc_casefold(raw_text) if store_normalized else ""
            )
            page_display = "-" if not (is_pdf or is_excel) else page_num
            entry = {
                "file": file_name,
                "path": path,
                "displayPath": display_path,
                "page": page_display,
                "pageRaw": page_num,
                "raw": raw_text,
                "norm": norm_base,
                "norm_jp": apply_space_mode(norm_base, "jp"),
                "norm_all": apply_space_mode(norm_base, "all"),
                "norm_strict": norm_strict,
                "norm_strict_jp": apply_space_mode(norm_strict, "jp"),
                "norm_strict_all": apply_space_mode(norm_strict, "all"),
                "folderId": folder_id,
                "folderName": folder_name,
                "fileId": file_id,
            }
            if store_normalized:
                entry.update(
                    {
                        "norm_cf": norm_cf_base,
                        "norm_cf_jp": apply_space_mode(norm_cf_base, "jp"),
                        "norm_cf_all": apply_space_mode(norm_cf_base, "all"),
                    }
                )
            pages.append(entry)
    return pages


def build_shared_blob(shared_path: Path, entries: List[Dict]) -> List[Dict]:
    store_normalized = env_bool("INDEX_STORE_NORMALIZED", True)
    offsets: List[Dict] = []
    total_size = 0
    for entry in entries:
        raw_b = entry["raw"].encode("utf-8")
        norm_b = encode_norm_text(entry["norm"])
        norm_jp_b = encode_norm_text(entry["norm_jp"])
        norm_all_b = encode_norm_text(entry["norm_all"])
        norm_strict_b = encode_norm_text(entry.get("norm_strict", ""))
        norm_strict_jp_b = encode_norm_text(entry.get("norm_strict_jp", ""))
        norm_strict_all_b = encode_norm_text(entry.get("norm_strict_all", ""))
        norm_cf_b = encode_norm_text(entry.get("norm_cf", "")) if store_normalized else b""
        norm_cf_jp_b = encode_norm_text(entry.get("norm_cf_jp", "")) if store_normalized else b""
        norm_cf_all_b = encode_norm_text(entry.get("norm_cf_all", "")) if store_normalized else b""
        entry_offsets = {
            "file": entry["file"],
            "path": entry["path"],
            "displayPath": entry["displayPath"],
            "page": entry["page"],
            "pageRaw": entry.get("pageRaw", entry["page"]),
            "folderId": entry["folderId"],
            "folderName": entry["folderName"],
            "fileId": entry.get("fileId") or file_id_from_path(entry["path"]),
            "raw_off": total_size,
            "raw_len": len(raw_b),
            "norm_off": total_size + len(raw_b),
            "norm_len": len(norm_b),
            "norm_jp_off": total_size + len(raw_b) + len(norm_b),
            "norm_jp_len": len(norm_jp_b),
            "norm_all_off": total_size + len(raw_b) + len(norm_b) + len(norm_jp_b),
            "norm_all_len": len(norm_all_b),
            "norm_strict_off": total_size + len(raw_b) + len(norm_b) + len(norm_jp_b) + len(norm_all_b),
            "norm_strict_len": len(norm_strict_b),
            "norm_strict_jp_off": total_size + len(raw_b) + len(norm_b) + len(norm_jp_b) + len(norm_all_b) + len(norm_strict_b),
            "norm_strict_jp_len": len(norm_strict_jp_b),
            "norm_strict_all_off": total_size + len(raw_b) + len(norm_b) + len(norm_jp_b) + len(norm_all_b) + len(norm_strict_b) + len(norm_strict_jp_b),
            "norm_strict_all_len": len(norm_strict_all_b),
        }
        extra_size = 0
        if store_normalized:
            entry_offsets.update(
                {
                    "norm_cf_off": total_size
                    + len(raw_b)
                    + len(norm_b)
                    + len(norm_jp_b)
                    + len(norm_all_b)
                    + len(norm_strict_b)
                    + len(norm_strict_jp_b)
                    + len(norm_strict_all_b),
                    "norm_cf_len": len(norm_cf_b),
                    "norm_cf_jp_off": total_size
                    + len(raw_b)
                    + len(norm_b)
                    + len(norm_jp_b)
                    + len(norm_all_b)
                    + len(norm_strict_b)
                    + len(norm_strict_jp_b)
                    + len(norm_strict_all_b)
                    + len(norm_cf_b),
                    "norm_cf_jp_len": len(norm_cf_jp_b),
                    "norm_cf_all_off": total_size
                    + len(raw_b)
                    + len(norm_b)
                    + len(norm_jp_b)
                    + len(norm_all_b)
                    + len(norm_strict_b)
                    + len(norm_strict_jp_b)
                    + len(norm_strict_all_b)
                    + len(norm_cf_b)
                    + len(norm_cf_jp_b),
                    "norm_cf_all_len": len(norm_cf_all_b),
                }
            )
            extra_size = len(norm_cf_b) + len(norm_cf_jp_b) + len(norm_cf_all_b)
        offsets.append(
            entry_offsets
        )
        total_size += (
            len(raw_b)
            + len(norm_b)
            + len(norm_jp_b)
            + len(norm_all_b)
            + len(norm_strict_b)
            + len(norm_strict_jp_b)
            + len(norm_strict_all_b)
            + extra_size
        )

    with open(shared_path, "wb") as f:
        f.truncate(total_size)
        for entry in entries:
            f.write(entry["raw"].encode("utf-8"))
            f.write(encode_norm_text(entry["norm"]))
            f.write(encode_norm_text(entry["norm_jp"]))
            f.write(encode_norm_text(entry["norm_all"]))
            f.write(encode_norm_text(entry.get("norm_strict", "")))
            f.write(encode_norm_text(entry.get("norm_strict_jp", "")))
            f.write(encode_norm_text(entry.get("norm_strict_all", "")))
            if store_normalized:
                f.write(encode_norm_text(entry.get("norm_cf", "")))
                f.write(encode_norm_text(entry.get("norm_cf_jp", "")))
                f.write(encode_norm_text(entry.get("norm_cf_all", "")))
    return offsets


def build_process_shared_store() -> List[Dict]:
    folder_snapshot: List[Dict] = []
    for folder_id, meta in folder_states.items():
        folder_path = meta["path"]
        folder_name = meta["name"]
        entries = memory_pages.get(folder_id, [])
        if not entries:
            log_info(f"共有ストア警告: 空のページ一覧 {folder_name}")
        shared_path = INDEX_DIR / f"{folder_id}_shared.bin"
        offsets = build_shared_blob(shared_path, entries)
        folder_snapshot.append(
            {
                "id": folder_id,
                "path": folder_path,
                "name": folder_name,
                "shared_path": str(shared_path),
                "entries": offsets,
            }
        )
    return folder_snapshot


# --- リクエストモデル ---
class SearchRequest(BaseModel):
    query: str = Field(..., description="検索キーワード（空白区切り）")
    mode: str = Field("AND", pattern="^(AND|OR)$")
    range_limit: int = Field(0, ge=0, le=5000)
    space_mode: str = Field("jp", pattern="^(none|jp|all)$")
    normalize_mode: str | None = Field(None)
    folders: List[str]

    @field_validator("query")
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("検索キーワードを指定してください")
        return v

    @field_validator("folders")
    def validate_folders(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("検索対象フォルダを1つ以上選択してください")
        return v

    @field_validator("normalize_mode")
    def validate_normalize_mode(cls, v: str | None) -> str | None:
        if v is None or not str(v).strip():
            return None
        mode = str(v).strip().lower()
        if mode not in {"exact", "normalized"}:
            raise ValueError("normalize_mode は exact または normalized を指定してください")
        return mode


class HeartbeatRequest(BaseModel):
    client_id: str = Field(..., min_length=8, max_length=64)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_folder_states()
    install_asyncio_exception_filter()
    global search_semaphore, rw_lock, search_worker_count, search_executor, search_execution_mode, search_concurrency, fixed_cache_index, memory_cache, query_stats, query_stats_dirty, query_stats_last_saved, folder_order
    rw_lock = AsyncRWLock()
    memory_cache = MemoryCache(
        DEFAULT_CACHE_MAX_ENTRIES,
        DEFAULT_CACHE_MAX_MB * 1024 * 1024,
        DEFAULT_CACHE_MAX_RESULT_KB,
    )
    init_cache_settings(memory_cache)
    fixed_cache_index = load_fixed_cache_index()
    # キャッシュ整合性検証（インデックス更新後の古いキャッシュを削除）
    fixed_cache_index = validate_cache_integrity(fixed_cache_index)
    save_fixed_cache_index(fixed_cache_index)
    prune_cache_dir(set(fixed_cache_index.keys()))
    query_stats = load_query_stats()
    query_stats_dirty = False
    query_stats_last_saved = time.time()
    folder_order = folder_order_map()
    budget, concurrency, workers = init_search_settings()
    search_worker_count = workers
    search_concurrency = concurrency
    search_semaphore = asyncio.Semaphore(concurrency)
    search_execution_mode = os.getenv("SEARCH_EXECUTION_MODE", "thread").strip().lower()
    loop = asyncio.get_running_loop()
    # インデックス構築はCPU負荷が高いのでスレッドで実施
    async with rw_lock.write_lock():
        await loop.run_in_executor(None, build_all_indexes)
    log_info(f"システムバージョン: {SYSTEM_VERSION} / インデックスバージョン: {INDEX_VERSION}")
    if search_execution_mode == "process":
        process_shared = os.getenv("SEARCH_PROCESS_SHARED", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        global PROCESS_SHARED
        PROCESS_SHARED = process_shared
        if process_shared:
            folder_snapshot = build_process_shared_store()
            search_executor = ProcessPoolExecutor(
                max_workers=concurrency,
                initializer=init_search_worker_shared,
                initargs=(folder_snapshot, host_aliases),
            )
            log_info(
                f"検索実行モード: process(shared) workers={concurrency} budget={budget}"
            )
        else:
            folder_snapshot = [
                {"id": fid, "path": meta["path"], "name": meta["name"]}
                for fid, meta in folder_states.items()
            ]
            search_executor = ProcessPoolExecutor(
                max_workers=concurrency,
                initializer=init_search_worker,
                initargs=(folder_snapshot, host_aliases),
            )
            log_info(f"検索実行モード: process workers={concurrency} budget={budget}")
    else:
        search_executor = None
        log_info(
            f"検索実行モード: thread concurrency={concurrency} workers={workers} budget={budget}"
        )
    sync_state()
    cert_dir = os.getenv("CERT_DIR", "certs")
    cert_path = (BASE_DIR / cert_dir).resolve()
    scheme = "https" if (cert_path / "lan-cert.pem").exists() else "http"
    urls = [
        f"{scheme}://{ip}:{os.getenv('PORT', '80')}"
        for ip in (get_ipv4_addresses() or ["0.0.0.0"])
    ]

    async def announce_access_urls():
        await asyncio.sleep(0.2)
        for url in urls:
            log_info(f"アクセスURL: {colorize_url(url)}")

    announce_task = asyncio.create_task(announce_access_urls())
    global last_search_ts
    last_search_ts = time.time()
    state.last_search_ts = last_search_ts
    warmup_task = None
    keepwarm_task = None
    # Warmup after build_all_indexes (WARMUP_ENABLED defaults to True)
    if env_bool("WARMUP_ENABLED", True) and is_primary_process():
        warmup_task = asyncio.create_task(warmup_startup_tasks())
        keepwarm_task = asyncio.create_task(warmup_keep_warm_loop(state))
    rebuild_fixed_cache()
    schedule_task = asyncio.create_task(schedule_index_rebuild())
    try:
        yield
    finally:
        for task in (announce_task, schedule_task, warmup_task, keepwarm_task):
            if task is None:
                continue
            task.cancel()
        await asyncio.gather(
            *(task for task in (announce_task, schedule_task, warmup_task, keepwarm_task) if task),
            return_exceptions=True,
        )
        if search_executor:
            search_executor.shutdown(wait=True)


# --- グローバル状態 ---
app = FastAPI(title="Preloaded Folder Search", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


config = load_config()
host_aliases = config.host_aliases
configured_folders = config.configured_folders
folder_states: Dict[str, Dict] = {}
memory_indexes: Dict[str, Dict[str, Dict]] = {}
memory_pages: Dict[str, List[Dict]] = {}
index_failures: Dict[str, Dict[str, str]] = {}
cache_lock = threading.RLock()
memory_cache: "MemoryCache | None" = None
query_stats: Dict[str, Dict] = {}
fixed_cache_index: Dict[str, Dict] = {}
fixed_cache_dirty = False
fixed_cache_last_saved = 0.0
query_stats_dirty = False
query_stats_last_saved = 0.0
fixed_cache_rebuild_in_progress = False
fixed_cache_last_trigger = 0.0
folder_order: Dict[str, int] = {}
access_urls: List[str] = []
PROCESS_SHARED = False
WORKER_SHARED_ENTRIES: Dict[str, List[Dict]] = {}
WORKER_SHARED_MM: Dict[str, mmap.mmap] = {}
WORKER_SHARED_FILES: Dict[str, object] = {}
search_semaphore: asyncio.Semaphore | None = None
rw_lock = None
search_worker_count = 1
search_concurrency = 1
search_executor = None
search_execution_mode = "thread"
normalize_mode_warning_emitted = False
file_id_map: Dict[str, Dict[str, str]] = {}
file_id_lock = threading.RLock()
active_client_lock = threading.Lock()
active_client_heartbeats: Dict[str, float] = {}
last_search_ts = 0.0

state = AppState(
    configured_folders=configured_folders,
    host_aliases=host_aliases,
    folder_states=folder_states,
    memory_indexes=memory_indexes,
    memory_pages=memory_pages,
    index_failures=index_failures,
    cache_lock=cache_lock,
    memory_cache=memory_cache,
    query_stats=query_stats,
    fixed_cache_index=fixed_cache_index,
    fixed_cache_dirty=fixed_cache_dirty,
    fixed_cache_last_saved=fixed_cache_last_saved,
    query_stats_dirty=query_stats_dirty,
    query_stats_last_saved=query_stats_last_saved,
    fixed_cache_rebuild_in_progress=fixed_cache_rebuild_in_progress,
    fixed_cache_last_trigger=fixed_cache_last_trigger,
    folder_order=folder_order,
    access_urls=access_urls,
    process_shared=PROCESS_SHARED,
    worker_shared_entries=WORKER_SHARED_ENTRIES,
    worker_shared_mm=WORKER_SHARED_MM,
    worker_shared_files=WORKER_SHARED_FILES,
    search_semaphore=search_semaphore,
    rw_lock=rw_lock,
    search_worker_count=search_worker_count,
    search_concurrency=search_concurrency,
    search_executor=search_executor,
    search_execution_mode=search_execution_mode,
    normalize_mode_warning_emitted=normalize_mode_warning_emitted,
    active_client_heartbeats=active_client_heartbeats,
    active_client_lock=active_client_lock,
)
ctx = AppContext(config=config, state=state)


def sync_state() -> None:
    state.memory_cache = memory_cache
    state.query_stats = query_stats
    state.fixed_cache_index = fixed_cache_index
    state.fixed_cache_dirty = fixed_cache_dirty
    state.fixed_cache_last_saved = fixed_cache_last_saved
    state.query_stats_dirty = query_stats_dirty
    state.query_stats_last_saved = query_stats_last_saved
    state.fixed_cache_rebuild_in_progress = fixed_cache_rebuild_in_progress
    state.fixed_cache_last_trigger = fixed_cache_last_trigger
    state.folder_order = folder_order
    state.process_shared = PROCESS_SHARED
    state.worker_shared_entries = WORKER_SHARED_ENTRIES
    state.worker_shared_mm = WORKER_SHARED_MM
    state.worker_shared_files = WORKER_SHARED_FILES
    state.search_semaphore = search_semaphore
    state.rw_lock = rw_lock
    state.search_worker_count = search_worker_count
    state.search_concurrency = search_concurrency
    state.search_executor = search_executor
    state.search_execution_mode = search_execution_mode
    state.normalize_mode_warning_emitted = normalize_mode_warning_emitted
    state.active_client_heartbeats = active_client_heartbeats


def create_app() -> FastAPI:
    app.state.ctx = ctx
    return app


WORKER_MEMORY_PAGES: Dict[str, List[Dict]] = {}
WORKER_FOLDER_META: Dict[str, Tuple[str, str]] = {}


@dataclass(frozen=True)
class SearchParams:
    mode: str
    range_limit: int
    space_mode: str
    normalize_mode: str


class AsyncRWLock:
    def __init__(self) -> None:
        self._readers = 0
        self._readers_lock = asyncio.Lock()
        self._writer_lock = asyncio.Lock()

    @asynccontextmanager
    async def read_lock(self):
        async with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                await self._writer_lock.acquire()
        try:
            yield
        finally:
            async with self._readers_lock:
                self._readers -= 1
                if self._readers == 0:
                    self._writer_lock.release()

    @asynccontextmanager
    async def write_lock(self):
        await self._writer_lock.acquire()
        try:
            yield
        finally:
            self._writer_lock.release()


def register_file_id(path: str, folder_id: str) -> str:
    file_id = file_id_from_path(path)
    with file_id_lock:
        existing = file_id_map.get(file_id)
        if existing and existing["path"] != path:
            log_warn(f"file_id衝突: {file_id} {existing['path']} != {path}")
        else:
            file_id_map[file_id] = {"path": path, "folderId": folder_id}
    return file_id


def rebuild_file_id_map() -> None:
    with file_id_lock:
        file_id_map.clear()
        for folder_id, cache in memory_indexes.items():
            for path in cache.keys():
                file_id = file_id_from_path(path)
                existing = file_id_map.get(file_id)
                if existing and existing["path"] != path:
                    log_warn(f"file_id衝突: {file_id} {existing['path']} != {path}")
                    continue
                file_id_map[file_id] = {"path": path, "folderId": folder_id}


def resolve_page_key(data: Dict, page: str) -> object | None:
    if page is None:
        return None
    page_str = str(page)
    if page_str.isdigit():
        page_int = int(page_str)
        if page_int in data:
            return page_int
    if page_str in data:
        return page_str
    return None


def describe_folder_state() -> List[Dict]:
    items = []
    for folder_id, meta in folder_states.items():
        files = memory_indexes.get(folder_id, {})
        display_path = display_path_for_path(meta["path"], host_aliases)
        items.append(
            {
                "id": folder_id,
                "path": meta["path"],
                "displayPath": display_path,
                "name": meta["name"],
                "ready": meta.get("ready", False),
                "message": meta.get("message", ""),
                "stats": meta.get("stats", {}),
                "indexed_files": len(files),
            }
        )
    return items


def init_folder_states():
    for folder in configured_folders:
        folder_path = folder["path"]
        folder_label = folder.get("label") or ""
        folder_id = folder_id_from_path(folder_path)
        display_path = display_path_for_path(folder_path, host_aliases)
        display_name = folder_label or Path(folder_path).name or folder_path
        folder_states[folder_id] = {
            "id": folder_id,
            "path": folder_path,
            "displayPath": display_path,
            "name": display_name,
            "ready": False,
            "message": "未処理",
            "stats": {},
        }


def migrate_legacy_indexes_to_generation() -> bool:
    """既存のフラットなインデックスを初回世代に移行."""
    import shutil

    # current.txt が既に存在する場合は移行不要
    if CURRENT_POINTER_FILE.exists():
        return False

    # 既存のフラットなインデックスファイルを検索
    legacy_indexes = list(INDEX_DIR.glob(f"index_*_{INDEX_VERSION}.pkl.gz"))
    if not legacy_indexes:
        return False

    log_info(f"既存インデックスの移行を開始: {len(legacy_indexes)} ファイル")

    # 初回世代を作成
    gen_name = create_generation_uuid()
    gen_dir = get_generation_dir(gen_name, build=False)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # インデックスファイルを移動
    migrated_count = 0
    for index_file in legacy_indexes:
        try:
            dest = gen_dir / index_file.name
            shutil.move(str(index_file), str(dest))
            migrated_count += 1
        except Exception as e:
            log_warn(f"インデックス移行失敗: {index_file.name} エラー={e}")

    if migrated_count > 0:
        # manifest.json を作成（簡易版）
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        manifest = {
            "index_uuid": gen_name,
            "schema_version": INDEX_VERSION,
            "created_at": now.isoformat(),
            "created_timestamp": int(now.timestamp()),
            "source_folders_hash": "",
            "folders": {},
            "migrated_from_legacy": True,
        }
        save_manifest(gen_dir, manifest)

        # currentポインターを設定
        set_current_generation_pointer(gen_name)
        log_info(f"既存インデックスの移行完了: {migrated_count} ファイル → gen_{gen_name}")
        return True

    return False


def build_all_indexes():
    """インデックス構築（世代ディレクトリ方式）."""
    import shutil

    log_info("インデックス構築開始")

    # 初回起動時: 既存のフラットなインデックスを世代ディレクトリに移行
    migrate_legacy_indexes_to_generation()

    # 新しい世代を作成
    gen_name = create_generation_uuid()
    build_gen_dir = get_generation_dir(gen_name, build=True)
    build_gen_dir.mkdir(parents=True, exist_ok=True)

    log_info(f"世代ディレクトリ構築: gen_{gen_name}")

    # エラー時にビルドディレクトリを確実に削除するため try-finally を使用
    try:
        # 各フォルダのインデックスを構築
        temp_indexes = {}
        temp_failures = {}

        for folder_id, meta in folder_states.items():
            path = meta["path"]
            if not os.path.isdir(path):
                meta["ready"] = False
                meta["message"] = f"フォルダが見つかりません: {path}"
                log_warn(f"フォルダが見つかりません: {path}")
                continue

            meta["message"] = "インデックス構築中..."
            log_info(f"フォルダ処理開始: {path}")

            # 差分更新のため、前回の世代ディレクトリを取得
            prev_gen_dir = get_current_generation_dir()

            prev_failures = index_failures.get(folder_id, {})
            if prev_gen_dir:
                prev_failures_disk, failures_loaded = load_failures(path, prev_gen_dir)
            else:
                prev_failures_disk, failures_loaded = {}, None
            merged_failures = dict(prev_failures_disk)
            merged_failures.update(prev_failures)
            cache, stats, failures = build_index_for_folder(
                path, merged_failures, build_gen_dir, prev_gen_dir, failures_loaded
            )

            temp_indexes[folder_id] = cache
            temp_failures[folder_id] = failures

            if os.getenv("SEARCH_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
                log_info(f"インデックス構築: {path} files={len(cache)}")

            meta["stats"] = stats
            log_info(f"フォルダ処理完了: {path}")

        # manifest.json を作成
        manifest = create_manifest(gen_name, build_gen_dir, folder_states)
        save_manifest(build_gen_dir, manifest)
        log_info(f"manifest.json 作成完了: gen_{gen_name}")

        # ビルドディレクトリを本番ディレクトリに移動（原子的）
        final_gen_dir = get_generation_dir(gen_name, build=False)
        build_gen_dir.replace(final_gen_dir)
        log_info(f"世代ディレクトリ移動完了: gen_{gen_name}")

        # currentポインターを更新（原子的）
        old_gen_name = get_current_generation_pointer()
        set_current_generation_pointer(gen_name)
        log_info(f"currentポインター更新: {old_gen_name or 'なし'} → gen_{gen_name}")

        # メモリ内のインデックスとページを更新
        for folder_id, cache in temp_indexes.items():
            memory_indexes[folder_id] = cache
            memory_pages[folder_id] = build_memory_pages(
                folder_id, folder_states[folder_id]["name"], cache
            )
            index_failures[folder_id] = temp_failures[folder_id]
            folder_states[folder_id]["ready"] = True
            folder_states[folder_id]["message"] = "準備完了"

            if os.getenv("SEARCH_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
                log_info(
                    f"ページ数: {folder_states[folder_id]['path']} "
                    f"files={len(cache)} pages={len(memory_pages[folder_id])}"
                )

        rebuild_file_id_map()

        # 古い世代をクリーンアップ
        grace_sec = env_int("INDEX_CLEANUP_GRACE_SEC", 300)
        cleanup_old_generations(gen_name, grace_sec)

        log_info("インデックス構築完了")

    except Exception as e:
        # エラー発生時: ビルドディレクトリを削除
        log_warn(f"インデックス構築失敗: {e}")
        if build_gen_dir.exists():
            try:
                shutil.rmtree(build_gen_dir)
                log_info(f"ビルドディレクトリ削除: {build_gen_dir}")
            except Exception as cleanup_error:
                log_warn(f"ビルドディレクトリ削除失敗: {cleanup_error}")
        raise


def install_asyncio_exception_filter():
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()

    def handler(loop, context):
        exc = context.get("exception")
        message = context.get("message", "")
        if isinstance(exc, ConnectionResetError):
            return
        if "ConnectionResetError" in message or "WinError 10054" in message:
            return
        if original_handler:
            original_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


def resolve_normalize_mode(value: str | None) -> str:
    global normalize_mode_warning_emitted
    store_normalized = env_bool("INDEX_STORE_NORMALIZED", True)
    if value:
        normalized = value.strip().lower()
        if normalized in {"exact", "normalized"}:
            if normalized == "normalized" and not store_normalized:
                if not normalize_mode_warning_emitted:
                    log_warn("正規化モードは INDEX_STORE_NORMALIZED=1 が必要なため exact にフォールバックします")
                    normalize_mode_warning_emitted = True
                return "exact"
            return normalized
    env_value = os.getenv("QUERY_NORMALIZE", "nfkc_casefold").strip().lower()
    if env_value == "nfkc_casefold":
        if store_normalized:
            return "normalized"
        if not normalize_mode_warning_emitted:
            log_warn("QUERY_NORMALIZE=nfkc_casefold には INDEX_STORE_NORMALIZED=1 が必要です")
            normalize_mode_warning_emitted = True
    return "exact"


def normalize_mode_label(mode: str) -> str:
    if mode == "normalized":
        return "ゆらぎ吸収"
    return "厳格（最小整形）"


class MemoryCache:
    def __init__(self, max_entries: int, max_bytes: int, max_result_kb: int) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.max_result_kb = max_result_kb
        self._data: OrderedDict[str, Dict] = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._total_bytes = 0
        self._lock = threading.RLock()

    def configure(self, max_entries: int, max_bytes: int, max_result_kb: int) -> None:
        with self._lock:
            self.max_entries = max_entries
            self.max_bytes = max_bytes
            self.max_result_kb = max_result_kb
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while self._data and (len(self._data) > self.max_entries or self._total_bytes > self.max_bytes):
            key, _ = self._data.popitem(last=False)
            size = self._sizes.pop(key, 0)
            self._total_bytes = max(0, self._total_bytes - size)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._sizes.clear()
            self._total_bytes = 0

    def get(self, key: str) -> Dict | None:
        with self._lock:
            payload = self._data.get(key)
            if payload is None:
                return None
            self._data.move_to_end(key)
            return payload

    def set(self, key: str, payload: Dict, size_bytes: int) -> None:
        size_kb = size_bytes // 1024
        if size_kb > self.max_result_kb:
            return
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                prev_size = self._sizes.get(key, 0)
                self._total_bytes = max(0, self._total_bytes - prev_size)
            self._data[key] = payload
            self._sizes[key] = size_bytes
            self._total_bytes += size_bytes
            self._evict_if_needed()

    def pop(self, key: str) -> None:
        with self._lock:
            if key in self._data:
                self._data.pop(key, None)
                size = self._sizes.pop(key, 0)
                self._total_bytes = max(0, self._total_bytes - size)


def normalize_query_key(query: str) -> str:
    return " ".join((query or "").strip().split())


def cache_key_for(query: str, params: "SearchParams", target_ids: List[str]) -> str:
    payload = {
        "v": SEARCH_CACHE_VERSION,
        "q": normalize_query_key(query),
        "mode": params.mode,
        "range": params.range_limit,
        "space": params.space_mode,
        "normalize": params.normalize_mode,
        "folders": sorted(target_ids),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def estimate_payload_bytes(payload: Dict) -> int:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return len(raw)


def normalize_folder_ids(ids: List[str]) -> List[str]:
    return sorted(ids)


def payload_matches_folders(payload: Dict, target_ids: List[str]) -> bool:
    expected = normalize_folder_ids(target_ids)
    stored = payload.get("folder_ids")
    if not isinstance(stored, list):
        return False
    return normalize_folder_ids(stored) == expected


def folder_order_map() -> Dict[str, int]:
    order: List[str] = []
    for folder in configured_folders:
        folder_path = folder.get("path")
        if not folder_path:
            continue
        order.append(folder_id_from_path(folder_path))
    return {fid: idx for idx, fid in enumerate(order)}


def apply_folder_order(payload: Dict, order_map: Dict[str, int]) -> Dict:
    results = payload.get("results")
    if not isinstance(results, list):
        return payload
    fallback = len(order_map) + 1
    results.sort(key=lambda r: order_map.get(r.get("folderId"), fallback))
    return payload


def load_fixed_cache_index() -> Dict[str, Dict]:
    if not FIXED_CACHE_INDEX.exists():
        return {}
    try:
        with open(FIXED_CACHE_INDEX, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def save_fixed_cache_index(index: Dict[str, Dict]) -> None:
    try:
        with open(FIXED_CACHE_INDEX, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    except Exception:
        return


def load_query_stats() -> Dict[str, Dict]:
    if not QUERY_STATS_PATH.exists():
        return {}
    try:
        with open(QUERY_STATS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def save_query_stats(stats: Dict[str, Dict]) -> None:
    try:
        with open(QUERY_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception:
        return


def touch_fixed_cache_entry(key: str) -> None:
    global fixed_cache_dirty
    entry = fixed_cache_index.get(key)
    if not entry:
        return
    entry["last_access"] = time.time()
    fixed_cache_dirty = True


def maybe_flush_fixed_cache_index(force: bool = False) -> None:
    global fixed_cache_dirty, fixed_cache_last_saved
    if not fixed_cache_dirty and not force:
        return
    now = time.time()
    if not force and now - fixed_cache_last_saved < 60:
        return
    save_fixed_cache_index(fixed_cache_index)
    fixed_cache_last_saved = now
    fixed_cache_dirty = False


def maybe_flush_query_stats(force: bool = False) -> None:
    global query_stats_dirty, query_stats_last_saved
    if not query_stats_dirty and not force:
        return
    now = time.time()
    interval = env_int("QUERY_STATS_FLUSH_SEC", 60)
    if not force and now - query_stats_last_saved < interval:
        return
    save_query_stats(query_stats)
    query_stats_last_saved = now
    query_stats_dirty = False


def fixed_cache_path(key: str, compressed: bool) -> Path:
    suffix = "json.gz" if compressed else "json"
    return CACHE_DIR / f"{key}.{suffix}"


def read_fixed_cache_payload(entry: Dict) -> Dict | None:
    path = entry.get("path")
    if not path:
        return None
    try:
        if entry.get("compressed"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_fixed_cache_payload(key: str, payload: Dict, compress: bool) -> Path | None:
    path = fixed_cache_path(key, compress)
    try:
        if compress:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        return path
    except Exception:
        return None


def prune_cache_dir(valid_keys: set[str]) -> None:
    """
    無効なキャッシュファイルを削除する。
    メタデータファイル（fixed_cache_index.json, query_stats.json）は保護される。
    """
    # メタデータファイルを除外（削除対象外）
    metadata_files = {"fixed_cache_index", "query_stats"}
    for file in CACHE_DIR.glob("*.json*"):
        stem = file.name.split(".")[0]
        # メタデータファイルは保護
        if stem in metadata_files:
            continue
        # 有効なキャッシュキーでなければ削除
        if stem not in valid_keys:
            try:
                file.unlink()
            except Exception:
                continue


def validate_cache_integrity(cache_index: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    キャッシュの整合性を検証し、不一致のエントリを削除する。
    起動時やインデックス更新後に呼び出される。
    """
    current_manifest = get_current_generation_manifest()
    if not current_manifest:
        log_warn("キャッシュ整合性検証: 現在の世代のmanifestが取得できません")
        return cache_index

    current_uuid = current_manifest.get("index_uuid")
    current_schema = current_manifest.get("schema_version")
    if not current_uuid or not current_schema:
        log_warn("キャッシュ整合性検証: manifestにindex_uuid/schema_versionがありません")
        return cache_index

    valid_cache: Dict[str, Dict] = {}
    invalid_count = 0

    for key, entry in cache_index.items():
        cached_uuid = entry.get("index_uuid")
        cached_schema = entry.get("schema_version")

        # index_uuid のチェック（欠落または不一致で無効化）
        if not cached_uuid or cached_uuid != current_uuid:
            log_info(
                f"キャッシュ削除（index_uuid不一致/欠落）: key={key[:8]}.. "
                f"cached={cached_uuid} current={current_uuid}"
            )
            invalid_count += 1
            continue

        # schema_version のチェック（欠落または不一致で無効化）
        if not cached_schema or cached_schema != current_schema:
            log_info(
                f"キャッシュ削除（schema不一致/欠落）: key={key[:8]}.. "
                f"cached={cached_schema} current={current_schema}"
            )
            invalid_count += 1
            continue

        # 有効なキャッシュとして保持
        valid_cache[key] = entry

    if invalid_count > 0:
        log_info(f"キャッシュ整合性検証完了: {invalid_count}件の古いキャッシュを削除しました")

    return valid_cache


def init_cache_settings(memory_cache: MemoryCache) -> None:
    max_entries = env_int("CACHE_MEM_MAX_ENTRIES", 200)
    max_mb = env_int("CACHE_MEM_MAX_MB", 200)
    max_result_kb = env_int("CACHE_MEM_MAX_RESULT_KB", 2000)
    memory_cache.configure(max_entries, max_mb * 1024 * 1024, max_result_kb)


def update_query_stats(
    stats: Dict[str, Dict],
    key: str,
    query: str,
    params: "SearchParams",
    target_ids: List[str],
    result_count: int,
    payload_kb: float,
    elapsed_ms: float | None,
    cached: bool,
) -> None:
    global query_stats_dirty
    now = time.time()
    entry = stats.get(key)
    if entry is None:
        entry = {
            "query": normalize_query_key(query),
            "mode": params.mode,
            "range": params.range_limit,
            "space": params.space_mode,
            "normalize": params.normalize_mode,
            "folders": sorted(target_ids),
            "count_total": 0,
            "count_uncached": 0,
            "total_time_ms": 0.0,
            "total_hits": 0,
            "total_payload_kb": 0.0,
            "last_access": now,
        }
    entry["count_total"] += 1
    entry["total_hits"] += result_count
    entry["total_payload_kb"] += payload_kb
    entry["last_access"] = now
    if not cached and elapsed_ms is not None:
        entry["count_uncached"] += 1
        entry["total_time_ms"] += elapsed_ms
    stats[key] = entry
    query_stats_dirty = True


def is_fixed_candidate(entry: Dict) -> bool:
    ttl_days = env_int("CACHE_FIXED_TTL_DAYS", 7)
    min_count = env_int("CACHE_FIXED_MIN_COUNT", 10)
    min_time = env_int("CACHE_FIXED_MIN_TIME_MS", 500)
    min_hits = env_int("CACHE_FIXED_MIN_HITS", 5000)
    min_kb = env_int("CACHE_FIXED_MIN_KB", 2000)
    now = time.time()
    if ttl_days > 0 and now - entry.get("last_access", 0) > ttl_days * 86400:
        return False
    if entry.get("count_total", 0) < min_count:
        return False
    avg_hits = entry.get("total_hits", 0) / max(1, entry.get("count_total", 1))
    avg_kb = entry.get("total_payload_kb", 0.0) / max(1, entry.get("count_total", 1))
    avg_time = entry.get("total_time_ms", 0.0) / max(1, entry.get("count_uncached", 1))
    if avg_time < min_time and avg_hits < min_hits and avg_kb < min_kb:
        return False
    return True


def rank_fixed_candidates(entries: List[Dict]) -> List[Dict]:
    def score(e: Dict) -> float:
        avg_time = e.get("total_time_ms", 0.0) / max(1, e.get("count_uncached", 1))
        avg_hits = e.get("total_hits", 0.0) / max(1, e.get("count_total", 1))
        return avg_time * max(1.0, avg_hits)
    return sorted(entries, key=score, reverse=True)


def prune_query_stats(stats: Dict[str, Dict]) -> None:
    ttl_days = env_int("QUERY_STATS_TTL_DAYS", 30)
    if ttl_days <= 0:
        return
    now = time.time()
    remove_keys = [k for k, v in stats.items() if now - v.get("last_access", 0) > ttl_days * 86400]
    for key in remove_keys:
        stats.pop(key, None)


def trigger_fixed_cache_rebuild(loop: asyncio.AbstractEventLoop, reason: str) -> None:
    global fixed_cache_rebuild_in_progress, fixed_cache_last_trigger
    cooldown = env_int("CACHE_FIXED_TRIGGER_COOLDOWN_SEC", 300)
    now = time.time()
    if fixed_cache_rebuild_in_progress:
        return
    if cooldown > 0 and now - fixed_cache_last_trigger < cooldown:
        return
    fixed_cache_rebuild_in_progress = True
    fixed_cache_last_trigger = now

    async def _run():
        nonlocal loop
        try:
            await loop.run_in_executor(None, rebuild_fixed_cache)
        finally:
            global fixed_cache_rebuild_in_progress
            fixed_cache_rebuild_in_progress = False

    log_info(f"固定キャッシュ再構築を起動します: {reason}")
    asyncio.create_task(_run())


def cpu_budget() -> int:
    env_budget = os.getenv("SEARCH_CPU_BUDGET", "").strip()
    if env_budget.isdigit():
        return max(1, int(env_budget))
    return max(1, os.cpu_count() or 4)


def total_worker_budget() -> int:
    env_budget = os.getenv("SEARCH_CPU_BUDGET", "").strip()
    if env_budget.isdigit():
        return max(1, int(env_budget))
    env_workers = os.getenv("SEARCH_WORKERS", "").strip()
    env_conc = os.getenv("SEARCH_CONCURRENCY", "").strip()
    if env_workers.isdigit() and env_conc.isdigit():
        return max(1, int(env_workers) * max(1, int(env_conc)))
    return max(1, os.cpu_count() or 4)


def heartbeat_ttl_sec() -> int:
    return max(1, env_int("HEARTBEAT_TTL_SEC", DEFAULT_HEARTBEAT_TTL_SEC))


def heartbeat_max_clients() -> int:
    env_max = os.getenv("HEARTBEAT_MAX_CLIENTS", "").strip()
    if env_max.isdigit():
        return max(1, int(env_max))
    return max(1, total_worker_budget())


def _prune_active_clients_locked(now: float, ttl_sec: int) -> None:
    expired = [cid for cid, last_seen in active_client_heartbeats.items() if now - last_seen > ttl_sec]
    for cid in expired:
        active_client_heartbeats.pop(cid, None)


def update_active_client(client_id: str) -> int:
    now = time.time()
    ttl_sec = heartbeat_ttl_sec()
    max_clients = heartbeat_max_clients()
    with active_client_lock:
        _prune_active_clients_locked(now, ttl_sec)
        if client_id in active_client_heartbeats:
            active_client_heartbeats[client_id] = now
        elif len(active_client_heartbeats) < max_clients:
            active_client_heartbeats[client_id] = now
        return len(active_client_heartbeats)


def get_active_client_count() -> int:
    now = time.time()
    ttl_sec = heartbeat_ttl_sec()
    max_clients = heartbeat_max_clients()
    with active_client_lock:
        _prune_active_clients_locked(now, ttl_sec)
        return min(len(active_client_heartbeats), max_clients)


def mark_search_activity() -> None:
    global last_search_ts
    last_search_ts = time.time()
    state.last_search_ts = last_search_ts


def replay_top_queries(reason: str) -> None:
    if not env_bool("QUERY_REPLAY_ENABLE", False):
        return
    if not is_primary_process():
        return
    top_k = env_int("QUERY_REPLAY_TOP_K", 10)
    if top_k <= 0:
        return
    prune_query_stats(query_stats)
    candidates = [v for v in query_stats.values() if v.get("query")]
    candidates.sort(key=lambda v: v.get("count_total", 0), reverse=True)
    if not candidates:
        log_notice(f"クエリ再生スキップ: 対象なし reason={reason}")
        return
    role = process_role()
    log_info(f"クエリ再生開始: reason={reason} role={role} top_k={top_k}")
    replayed = 0
    for entry in candidates[:top_k]:
        target_ids = [
            fid
            for fid in entry.get("folders", [])
            if fid in folder_states and folder_states[fid].get("ready")
        ]
        if not target_ids:
            continue
        params = SearchParams(
            entry.get("mode", "AND"),
            int(entry.get("range", 0)),
            entry.get("space", "jp"),
            resolve_normalize_mode(entry.get("normalize")),
        )
        try:
            run_search_direct(entry.get("query", ""), params, target_ids, False)
            replayed += 1
        except Exception:
            continue
    log_info(f"クエリ再生完了: reason={reason} role={role} replayed={replayed}")


async def warmup_startup_tasks() -> None:
    """Startup warmup tasks: warmup current generation and replay top queries."""
    gen_name = get_current_generation_pointer()
    if gen_name:
        await warmup_trigger_warmup("startup", gen_name)
        state.last_warmup_ts = time.time()
    replay_top_queries("startup")


def init_search_settings() -> Tuple[int, int, int]:
    budget = total_worker_budget()
    env_conc = os.getenv("SEARCH_CONCURRENCY", "").strip()
    concurrency = budget
    if env_conc.isdigit():
        concurrency = max(1, int(env_conc))
    env_workers = os.getenv("SEARCH_WORKERS", "").strip()
    if env_workers.isdigit():
        workers = max(1, int(env_workers))
    else:
        workers = max(1, budget // max(1, concurrency))
    return budget, concurrency, workers


def parse_rebuild_schedule(value: str) -> Tuple[str, int, int]:
    raw = (value or "").strip()
    if not raw:
        return ("", 0, 0)
    if raw.lower().endswith("h"):
        try:
            hours = int(raw[:-1])
            return ("interval", hours, 0)
        except ValueError:
            return ("", 0, 0)
    if raw.isdigit():
        return ("interval", int(raw), 0)
    try:
        parts = raw.split(":")
        if len(parts) == 2:
            hour = int(parts[0])
            minute = int(parts[1])
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return ("time", hour, minute)
    except ValueError:
        return ("", 0, 0)
    return ("", 0, 0)


async def schedule_index_rebuild():
    schedule = os.getenv("REBUILD_SCHEDULE", "").strip()
    mode, a, b = parse_rebuild_schedule(schedule)
    if not mode:
        return
    log_info(f"インデックス再構築スケジュール有効: {schedule}")
    while True:
        if mode == "interval":
            await asyncio.sleep(max(1, a) * 3600)
        elif mode == "time":
            now = time.localtime()
            target = time.mktime(
                (now.tm_year, now.tm_mon, now.tm_mday, a, b, 0, now.tm_wday, now.tm_yday, now.tm_isdst)
            )
            if target <= time.mktime(now):
                target += 86400
            await asyncio.sleep(max(1, int(target - time.mktime(now))))
        else:
            return
        if rw_lock is None:
            continue
        log_info("スケジュール再構築開始")
        async with rw_lock.write_lock():
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, build_all_indexes)
            if search_execution_mode == "process":
                global search_executor
                if search_executor:
                    search_executor.shutdown(wait=True)
                if PROCESS_SHARED:
                    folder_snapshot = build_process_shared_store()
                    search_executor = ProcessPoolExecutor(
                        max_workers=search_concurrency,
                        initializer=init_search_worker_shared,
                        initargs=(folder_snapshot, host_aliases),
                    )
                    log_info("スケジュール再構築: process(shared) を再初期化")
                else:
                    folder_snapshot = [
                        {"id": fid, "path": meta["path"], "name": meta["name"]}
                        for fid, meta in folder_states.items()
                    ]
                    search_executor = ProcessPoolExecutor(
                        max_workers=search_concurrency,
                        initializer=init_search_worker,
                        initargs=(folder_snapshot, host_aliases),
                    )
                    log_info("スケジュール再構築: process を再初期化")
            rebuild_fixed_cache()
            sync_state()
        # Warmup after generation switch (WARMUP_ENABLED defaults to True)
        if env_bool("WARMUP_ENABLED", True):
            gen_name = get_current_generation_pointer()
            if gen_name:
                await warmup_trigger_warmup("generation_switch", gen_name)
                state.last_warmup_ts = time.time()
            replay_top_queries("generation_switch")
        # Cleanup old warmup lock files
        cleanup_old_warmup_locks()
        log_info("スケジュール再構築完了")


def rebuild_fixed_cache():
    global fixed_cache_index

    # 現在の世代のmanifestを取得（キャッシュ整合性のため）
    current_manifest = get_current_generation_manifest()
    if not current_manifest:
        log_warn("固定キャッシュ再構築: 現在の世代のmanifestが取得できません")
        return

    current_index_uuid = current_manifest.get("index_uuid")
    current_schema_version = current_manifest.get("schema_version")
    if not current_index_uuid or not current_schema_version:
        log_warn("固定キャッシュ再構築: manifestにindex_uuid/schema_versionがありません")
        return

    log_info(f"固定キャッシュ再構築開始: index_uuid={current_index_uuid}, schema={current_schema_version}")

    prune_query_stats(query_stats)
    candidates = [v for v in query_stats.values() if is_fixed_candidate(v)]
    candidates = rank_fixed_candidates(candidates)
    max_entries = env_int("CACHE_FIXED_MAX_ENTRIES", 20)
    candidates = candidates[:max_entries]

    source_entries: List[Dict] = candidates
    if not source_entries and fixed_cache_index:
        ttl_days = env_int("CACHE_FIXED_TTL_DAYS", 7)
        now = time.time()
        source_entries = []
        for entry in fixed_cache_index.values():
            last_access = entry.get("last_access", 0)
            if ttl_days > 0 and now - last_access > ttl_days * 86400:
                continue
            source_entries.append(entry)

    new_index: Dict[str, Dict] = {}
    for entry in source_entries:
        target_ids = [fid for fid in entry.get("folders", []) if fid in folder_states and folder_states[fid].get("ready")]
        if not target_ids:
            continue
        normalize_mode = resolve_normalize_mode(entry.get("normalize"))
        params = SearchParams(entry["mode"], entry["range"], entry["space"], normalize_mode)
        try:
            results, keywords = run_search_direct(entry["query"], params, target_ids)
        except Exception:
            continue
        payload = {
            "count": len(results),
            "results": results,
            "keywords": keywords,
            "folder_ids": normalize_folder_ids(target_ids),
        }
        payload = apply_folder_order(payload, folder_order)
        size_bytes = estimate_payload_bytes(payload)
        size_kb = size_bytes / 1024
        compress = size_kb >= env_int("CACHE_COMPRESS_MIN_KB", 2000)
        key = cache_key_for(entry["query"], params, target_ids)
        path = write_fixed_cache_payload(key, payload, compress)
        if not path:
            continue
        new_index[key] = {
            "path": str(path),
            "compressed": compress,
            "payload_kb": size_kb,
            "result_count": len(results),
            "last_access": entry.get("last_access", time.time()),
            "last_built": time.time(),
            "query": entry["query"],
            "mode": entry["mode"],
            "range": entry["range"],
            "space": entry["space"],
            "normalize": normalize_mode,
            "folders": target_ids,
            "index_uuid": current_index_uuid,
            "schema_version": current_schema_version,
        }

    with cache_lock:
        fixed_cache_index = new_index
        save_fixed_cache_index(fixed_cache_index)
        if memory_cache:
            memory_cache.clear()
        global fixed_cache_dirty, fixed_cache_last_saved
        fixed_cache_dirty = False
        fixed_cache_last_saved = time.time()
        maybe_flush_query_stats(force=True)
        sync_state()
    prune_cache_dir(set(new_index.keys()))


def per_request_workers() -> int:
    budget = total_worker_budget()
    active_clients = get_active_client_count()
    env_workers = os.getenv("SEARCH_WORKERS", "").strip()
    max_per_request = budget
    if env_workers.isdigit():
        max_per_request = max(1, int(env_workers))
    workers = budget // max(1, active_clients)
    workers = max(1, min(workers, max_per_request))
    return workers


@app.get("/", response_class=FileResponse)
async def read_root():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UIが見つかりません")
    return FileResponse(index_file)


@app.get("/api/folders")
async def get_folders():
    return JSONResponse({"folders": describe_folder_state()})


@app.post("/api/heartbeat")
async def heartbeat(req: HeartbeatRequest):
    active_clients = update_active_client(req.client_id)
    return JSONResponse(
        {
            "status": "ok",
            "active_clients": active_clients,
            "ttl_sec": heartbeat_ttl_sec(),
        }
    )


@app.get("/api/folders/{folder_id}/files")
async def get_folder_files(folder_id: str, scope: str = "indexed"):
    if folder_id not in folder_states:
        raise HTTPException(status_code=404, detail="フォルダが見つかりません")
    meta = folder_states[folder_id]
    if not meta.get("ready"):
        raise HTTPException(status_code=400, detail="このフォルダは準備中です")

    base_path = meta["path"]
    cache = memory_indexes.get(folder_id, {})
    failure_map = index_failures.get(folder_id, {})
    scope = (scope or "indexed").strip().lower()
    if scope not in {"indexed", "missing", "all"}:
        raise HTTPException(status_code=400, detail="scopeが不正です")

    indexed_paths = set(cache.keys())
    files = []
    if scope == "indexed":
        for path in sorted(indexed_paths):
            rel = os.path.relpath(path, base_path)
            files.append({"path": path, "relative": rel, "indexed": True, "reason": ""})
    else:
        all_files = scan_files(base_path)
        for path_obj in sorted(all_files):
            path = str(path_obj)
            is_indexed = path in indexed_paths
            if scope == "missing" and is_indexed:
                continue
            reason = ""
            if not is_indexed:
                reason = failure_map.get(path, "理由不明")
            rel = os.path.relpath(path, base_path)
            files.append({"path": path, "relative": rel, "indexed": is_indexed, "reason": reason})
    return {"folder": folder_id, "name": meta["name"], "files": files, "scope": scope}


@app.get("/api/detail")
async def get_detail(
    file_id: str = Query(..., min_length=12, max_length=12, pattern=r"^[0-9a-f]{12}$"),
    page: str = Query(..., min_length=1),
    hit_pos: int = Query(0, ge=0),
):
    global rw_lock
    if rw_lock is None:
        raise HTTPException(status_code=503, detail="検索システムの初期化中です")
    async with rw_lock.read_lock():
        with file_id_lock:
            info = file_id_map.get(file_id)
        if not info:
            raise HTTPException(status_code=404, detail="ファイルが見つかりません")
        folder_id = info["folderId"]
        if folder_id not in folder_states or not folder_states[folder_id].get("ready"):
            raise HTTPException(status_code=400, detail="対象フォルダが準備中です")
        cache = memory_indexes.get(folder_id, {})
        meta = cache.get(info["path"])
        if not meta:
            raise HTTPException(status_code=404, detail="ファイルが見つかりません")
        data = meta.get("data", {})
        page_key = resolve_page_key(data, page)
        if page_key is None:
            raise HTTPException(status_code=404, detail="ページが見つかりません")
        raw_text = data.get(page_key)
        detail_text = build_detail_text(raw_text or "", hit_pos)
        return {"file_id": file_id, "page": page_key, "detail": detail_text}


def _try_get_memory_cache(
    cache_key: str,
    target_ids: List[str],
    req_query: str,
    params: "SearchParams",
) -> Dict | None:
    """メモリキャッシュから結果を取得。ヒットした場合は統計を更新して返す。"""
    with cache_lock:
        cached_payload = memory_cache.get(cache_key) if memory_cache else None
    if cached_payload and payload_matches_folders(cached_payload, target_ids):
        cached_payload = apply_folder_order(cached_payload, folder_order)
        cached_bytes = estimate_payload_bytes(cached_payload)
        cached_kb = cached_bytes / 1024
        update_query_stats(
            query_stats,
            cache_key,
            req_query,
            params,
            target_ids,
            cached_payload.get("count", 0),
            cached_kb,
            None,
            True,
        )
        maybe_flush_query_stats()
        return cached_payload
    elif cached_payload:
        with cache_lock:
            if memory_cache:
                memory_cache.pop(cache_key)
    return None


def _try_get_fixed_cache(
    cache_key: str,
    target_ids: List[str],
    req_query: str,
    params: "SearchParams",
) -> Dict | None:
    """固定キャッシュから結果を取得。ヒットした場合はメモリキャッシュにも保存して返す。"""
    with cache_lock:
        fixed_entry = fixed_cache_index.get(cache_key)
    if not fixed_entry:
        return None

    # キャッシュ整合性チェック: index_uuid と schema_version を検証
    current_manifest = get_current_generation_manifest()
    if current_manifest:
        current_uuid = current_manifest.get("index_uuid")
        current_schema = current_manifest.get("schema_version")
        cached_uuid = fixed_entry.get("index_uuid")
        cached_schema = fixed_entry.get("schema_version")

        # index_uuid のチェック（欠落または不一致で無効化）
        if current_uuid and (not cached_uuid or cached_uuid != current_uuid):
            log_warn(
                f"キャッシュ無効化（index_uuid不一致/欠落）: key={cache_key[:8]}.. "
                f"cached={cached_uuid} current={current_uuid}"
            )
            with cache_lock:
                fixed_cache_index.pop(cache_key, None)
                save_fixed_cache_index(fixed_cache_index)
            return None

        # schema_version のチェック（欠落または不一致で無効化）
        if current_schema and (not cached_schema or cached_schema != current_schema):
            log_warn(
                f"キャッシュ無効化（schema不一致/欠落）: key={cache_key[:8]}.. "
                f"cached={cached_schema} current={current_schema}"
            )
            with cache_lock:
                fixed_cache_index.pop(cache_key, None)
                save_fixed_cache_index(fixed_cache_index)
            return None

    payload = read_fixed_cache_payload(fixed_entry)
    if payload and payload_matches_folders(payload, target_ids):
        payload = apply_folder_order(payload, folder_order)
        cached_kb = fixed_entry.get("payload_kb")
        if cached_kb is None:
            cached_kb = estimate_payload_bytes(payload) / 1024
        update_query_stats(
            query_stats,
            cache_key,
            req_query,
            params,
            target_ids,
            payload.get("count", 0),
            cached_kb,
            None,
            True,
        )
        with cache_lock:
            if memory_cache:
                memory_cache.set(cache_key, payload, int(cached_kb * 1024))
            touch_fixed_cache_entry(cache_key)
        maybe_flush_fixed_cache_index()
        maybe_flush_query_stats()
        return payload
    with cache_lock:
        fixed_cache_index.pop(cache_key, None)
        save_fixed_cache_index(fixed_cache_index)
    return None


@app.post("/api/search")
async def search(req: SearchRequest):
    global rw_lock, search_semaphore, search_executor, search_execution_mode
    if rw_lock is None or search_semaphore is None:
        raise HTTPException(status_code=503, detail="検索システムの初期化中です")

    # Normal search (full index scan)
    async with search_semaphore:
        async with rw_lock.read_lock():
            start_time = time.time()
            available_ids = set(folder_states.keys())
            target_ids = [f for f in req.folders if f in available_ids and folder_states[f].get("ready")]
            if not target_ids:
                raise HTTPException(status_code=400, detail="有効な検索対象フォルダがありません")

            normalize_mode = resolve_normalize_mode(req.normalize_mode)
            params = SearchParams(req.mode, req.range_limit, req.space_mode, normalize_mode)
            cache_key = cache_key_for(req.query, params, target_ids)
            mark_search_activity()

            # キャッシュからの取得を試行
            cached_result = _try_get_memory_cache(cache_key, target_ids, req.query, params)
            if cached_result:
                return cached_result
            cached_result = _try_get_fixed_cache(cache_key, target_ids, req.query, params)
            if cached_result:
                return cached_result

            keywords, norm_keyword_groups = build_query_groups(
                req.query,
                req.space_mode,
                normalize_mode,
            )
            raw_keywords = keywords
            keywords_for_response = keywords
            worker_count = per_request_workers()
            loop = asyncio.get_running_loop()
            if os.getenv("SEARCH_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
                folder_debug = []
                for fid in target_ids:
                    name = folder_states[fid]["name"]
                    entries = memory_pages.get(fid)
                    folder_debug.append(f"{name}({fid}) entries={len(entries) if entries is not None else 'none'}")
                log_info(f"検索デバッグ: query='{req.query}' targets={folder_debug}")
            if search_execution_mode == "process":
                results = await loop.run_in_executor(
                    search_executor,
                    perform_search_process,
                    target_ids,
                    params,
                    norm_keyword_groups,
                    raw_keywords,
                    worker_count,
                    False,
                )
            else:
                results = await loop.run_in_executor(
                    None,
                    perform_search,
                    target_ids,
                    params,
                    norm_keyword_groups,
                    raw_keywords,
                    worker_count,
                    False,
                )
        elapsed = time.time() - start_time
        log_info(
            f"検索完了: folders={len(target_ids)} results={len(results)} "
            f"mode={req.mode} normalize={normalize_mode} workers={worker_count} "
            f"active_clients={get_active_client_count()} 時間={elapsed:.2f}s"
        )

        # Get current index UUID
        current_gen_name = get_current_generation_pointer()
        index_uuid = current_gen_name or "unknown"

        payload = {
            "count": len(results),
            "results": results,
            "keywords": keywords_for_response,
            "folder_ids": normalize_folder_ids(target_ids),
            "index_uuid": index_uuid,
            "normalize_mode": normalize_mode,
        }
        payload = apply_folder_order(payload, folder_order)
        payload_bytes = estimate_payload_bytes(payload)
        payload_kb = payload_bytes / 1024
        update_query_stats(
            query_stats,
            cache_key,
            req.query,
            params,
            target_ids,
            len(results),
            payload_kb,
            elapsed * 1000,
            False,
        )
        with cache_lock:
            if memory_cache:
                memory_cache.set(cache_key, payload, payload_bytes)
        maybe_flush_query_stats()

        if is_fixed_candidate(query_stats.get(cache_key, {})) and cache_key not in fixed_cache_index:
            trigger_fixed_cache_rebuild(loop, "クエリ条件到達")

        # UI側でハイライトしやすいように検索キーワードも返す
        return payload


@app.post("/api/export")
async def export_results(req: SearchRequest, format: str = "csv"):
    """検索結果をCSV/JSON形式でエクスポートする"""
    global rw_lock, search_semaphore, search_executor, search_execution_mode
    if rw_lock is None or search_semaphore is None:
        raise HTTPException(status_code=503, detail="検索システムの初期化中です")

    # Validate format
    if format not in ["csv", "json"]:
        raise HTTPException(status_code=400, detail="format は csv または json を指定してください")

    # Execute search first
    async with search_semaphore:
        async with rw_lock.read_lock():
            available_ids = set(folder_states.keys())
            target_ids = [f for f in req.folders if f in available_ids and folder_states[f].get("ready")]
            if not target_ids:
                raise HTTPException(status_code=400, detail="有効な検索対象フォルダがありません")

            normalize_mode = resolve_normalize_mode(req.normalize_mode)
            params = SearchParams(req.mode, req.range_limit, req.space_mode, normalize_mode)
            mark_search_activity()
            keywords, norm_keyword_groups = build_query_groups(
                req.query,
                req.space_mode,
                normalize_mode,
            )
            raw_keywords = keywords
            worker_count = per_request_workers()
            loop = asyncio.get_running_loop()

            if search_execution_mode == "process":
                results = await loop.run_in_executor(
                    search_executor,
                    perform_search_process,
                    target_ids,
                    params,
                    norm_keyword_groups,
                    raw_keywords,
                    worker_count,
                    True,
                )
            else:
                results = await loop.run_in_executor(
                    None,
                    perform_search,
                    target_ids,
                    params,
                    norm_keyword_groups,
                    raw_keywords,
                    worker_count,
                    True,
                )

    # Get current index UUID
    current_gen_name = get_current_generation_pointer()
    index_uuid = current_gen_name or "unknown"

    # Prepare metadata
    metadata = {
        "query": req.query,
        "mode": req.mode,
        "range_limit": req.range_limit,
        "space_mode": req.space_mode,
        "normalize_mode": normalize_mode,
        "folders": target_ids,
        "folder_names": [folder_states[fid]["name"] for fid in target_ids if fid in folder_states],
        "index_uuid": index_uuid,
        "result_count": len(results),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if format == "csv":
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Write metadata as comments
        writer.writerow(["# フォルダ内テキスト検索 - エクスポート"])
        writer.writerow(["# クエリ", req.query])
        writer.writerow(["# モード", req.mode])
        writer.writerow(["# 範囲", req.range_limit])
        writer.writerow(["# 空白除去", req.space_mode])
        writer.writerow(["# 表記ゆれ", normalize_mode_label(normalize_mode)])
        writer.writerow(["# インデックスUUID", index_uuid])
        writer.writerow(["# エクスポート日時", metadata["exported_at"]])
        writer.writerow(["# 検索フォルダ", ", ".join(metadata["folder_names"])])
        writer.writerow(["# 結果件数", len(results)])
        writer.writerow([])

        # Write header
        writer.writerow(["ファイル名", "パス", "ページ", "フォルダ", "スニペット", "詳細"])

        # Write results
        for result in results:
            writer.writerow([
                result.get("file", ""),
                result.get("displayPath") or result.get("path", ""),
                result.get("page", ""),
                result.get("folderName", ""),
                result.get("context", ""),
                result.get("detail", ""),
            ])

        csv_content = output.getvalue()
        output.close()

        # Return as downloadable file
        return StreamingResponse(
            io.BytesIO(csv_content.encode("utf-8-sig")),  # UTF-8 with BOM for Excel compatibility
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="search_results_{time.strftime("%Y%m%d_%H%M%S")}.csv"'
            }
        )

    else:  # json
        # Create JSON with metadata
        export_data = {
            "metadata": metadata,
            "results": results,
        }

        json_content = json.dumps(export_data, ensure_ascii=False, indent=2)

        return StreamingResponse(
            io.BytesIO(json_content.encode("utf-8")),
            media_type="application/json; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="search_results_{time.strftime("%Y%m%d_%H%M%S")}.json"'
            }
        )


@app.get("/api/health")
async def health():
    ready = all(m.get("ready") for m in folder_states.values()) if folder_states else False
    return JSONResponse(
        {"status": "ok", "ready": ready},
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# --- 実行ヘルパー ---
def run():
    """ローカル開発用エントリポイント."""
    import uvicorn
    import signal

    port = int(os.getenv("PORT", "80"))
    cert_dir = os.getenv("CERT_DIR", "certs")
    cert_path = (BASE_DIR / cert_dir).resolve()
    cert_file = cert_path / "lan-cert.pem"
    key_file = cert_path / "lan-key.pem"

    ssl_kwargs = {}
    if cert_file.exists() and key_file.exists():
        ssl_kwargs = {
            "ssl_certfile": str(cert_file),
            "ssl_keyfile": str(key_file),
        }

    config = uvicorn.Config(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        **ssl_kwargs,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    interrupt_count = {"count": 0}

    def handle_interrupt(signum, frame):
        interrupt_count["count"] += 1
        if interrupt_count["count"] == 1:
            log_info("Ctrl+Cを検知しました。もう一度Ctrl+Cで強制終了します。")
            server.should_exit = True
            return
        log_info("Ctrl+Cを再度検知しました。強制終了します。")
        os._exit(1)

    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        signal.signal(signal.SIGTERM, handle_interrupt)
    except Exception:
        pass

    server.run()


if __name__ == "__main__":
    run()
