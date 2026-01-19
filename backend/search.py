"""Search logic for worker processes and common helpers."""
from __future__ import annotations

import mmap
import os
import signal
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from .config import (
    SEARCH_ENTRIES_CHUNK_THRESHOLD,
    SNIPPET_PREFIX_CHARS,
    SNIPPET_TOTAL_LENGTH,
    display_path_for_path,
)
from .index_ops import load_index_from_disk
from .text_utils import (
    apply_space_mode,
    build_detail_text,
    build_flexible_keyword_regex,
    encode_norm_text,
    normalize_invisible_separators,
    normalize_snippet_text,
    normalize_text,
    normalize_text_minimal,
    normalize_text_nfkc_casefold,
)
from .utils import file_id_from_path, log_info

SYSTEM_VERSION = "1.3.1"

# Worker process globals
WORKER_MEMORY_PAGES: Dict[str, List[Dict]] = {}
WORKER_FOLDER_META: Dict[str, Tuple[str, str]] = {}
WORKER_SHARED_ENTRIES: Dict[str, List[Dict]] = {}
WORKER_SHARED_MM: Dict[str, mmap.mmap] = {}
WORKER_SHARED_FILES: Dict[str, object] = {}
PROCESS_SHARED = False
host_aliases: Dict[str, str] = {}


def _find_raw_hit_position(
    raw_for_positions: str,
    raw_keywords: List[str],
    space_mode: str,
) -> int:
    """Find keyword position in raw text."""
    raw_hit_pos = -1
    raw_lower = raw_for_positions.lower()
    for kw in raw_keywords:
        if not kw:
            continue
        raw_idx = raw_lower.find(kw.lower())
        if raw_idx != -1:
            raw_hit_pos = raw_idx
            break

    if raw_hit_pos == -1 and space_mode != "none":
        for kw in raw_keywords:
            if not kw:
                continue
            flex = build_flexible_keyword_regex(kw)
            if not flex:
                continue
            match = flex.search(raw_for_positions)
            if match:
                raw_hit_pos = match.start()
                break

    return raw_hit_pos


def _build_search_result(
    entry: Dict,
    raw_text: str,
    raw_for_positions: str,
    raw_hit_pos: int,
    include_detail: bool,
) -> Dict:
    """Build search result with snippet and detail."""
    snippet_start = max(0, raw_hit_pos - SNIPPET_PREFIX_CHARS) if raw_hit_pos != -1 else 0
    snippet_end = min(len(raw_for_positions), snippet_start + SNIPPET_TOTAL_LENGTH)
    snippet_raw = raw_for_positions[snippet_start:snippet_end]
    snippet = f"...{normalize_snippet_text(snippet_raw)}..."

    file_id = entry.get("fileId") or file_id_from_path(entry["path"])
    page_raw = entry.get("pageRaw", entry.get("page"))
    hit_pos = raw_hit_pos if raw_hit_pos >= 0 else 0
    result = {
        "file": entry["file"],
        "path": entry["path"],
        "displayPath": entry.get(
            "displayPath", display_path_for_path(entry["path"], host_aliases)
        ),
        "page": entry["page"],
        "context": snippet,
        "detail_key": {
            "file_id": file_id,
            "page": page_raw,
            "hit_pos": hit_pos,
        },
        "folderId": entry["folderId"],
        "folderName": entry["folderName"],
    }
    if include_detail:
        result["detail"] = build_detail_text(raw_text, hit_pos)
    return result


def search_text_logic(
    entry: Dict,
    norm_keyword_groups: List[List[str]],
    raw_keywords: List[str],
    search_mode: str,
    range_limit: int,
    space_mode: str,
    normalize_mode: str,
    include_detail: bool,
) -> List[Dict]:
    if not norm_keyword_groups:
        return []

    raw_text = entry["raw"]
    raw_for_positions = normalize_invisible_separators(raw_text)
    if normalize_mode == "normalized":
        if space_mode == "all":
            norm_text = entry.get("norm_cf_all") or entry["norm_all"]
        elif space_mode == "jp":
            norm_text = entry.get("norm_cf_jp") or entry["norm_jp"]
        else:
            norm_text = entry.get("norm_cf") or entry["norm"]
    else:
        if space_mode == "all":
            norm_text = entry.get("norm_strict_all") or entry["norm_all"]
        elif space_mode == "jp":
            norm_text = entry.get("norm_strict_jp") or entry["norm_jp"]
        else:
            norm_text = entry.get("norm_strict") or entry["norm"]
    is_match = False
    match_pos = -1

    if search_mode == "OR":
        for group in norm_keyword_groups:
            for k in group:
                if not k:
                    continue
                idx = norm_text.find(k)
                if idx != -1:
                    is_match = True
                    match_pos = idx
                    break
            if is_match:
                break
    elif search_mode == "AND":
        for group in norm_keyword_groups:
            if not any(k in norm_text for k in group if k):
                return []
        if range_limit == 0 or len(norm_keyword_groups) == 1:
            first_group = norm_keyword_groups[0]
            first_kw = next((k for k in first_group if k), "")
            is_match = True
            match_pos = norm_text.find(first_kw) if first_kw else 0
        else:
            positions = []
            for group_id, group in enumerate(norm_keyword_groups):
                for kw in group:
                    if not kw:
                        continue
                    start = 0
                    while True:
                        idx = norm_text.find(kw, start)
                        if idx == -1:
                            break
                        positions.append((idx, group_id))
                        start = idx + 1
            positions.sort()
            left = 0
            count_map = defaultdict(int)
            unique_count = 0
            target_unique = len(norm_keyword_groups)
            min_window_len = float("inf")
            best_start_pos = -1
            for right in range(len(positions)):
                r_idx, r_group_id = positions[right]
                if count_map[r_group_id] == 0:
                    unique_count += 1
                count_map[r_group_id] += 1
                while unique_count == target_unique:
                    l_idx, l_group_id = positions[left]
                    current_len = r_idx - l_idx
                    if current_len <= range_limit:
                        is_match = True
                        if current_len < min_window_len:
                            min_window_len = current_len
                            best_start_pos = l_idx
                    count_map[l_group_id] -= 1
                    if count_map[l_group_id] == 0:
                        unique_count -= 1
                    left += 1
            if is_match and best_start_pos != -1:
                match_pos = best_start_pos
        if not is_match:
            return []

    if not is_match:
        return []

    raw_hit_pos = _find_raw_hit_position(raw_for_positions, raw_keywords, space_mode)
    return [_build_search_result(entry, raw_text, raw_for_positions, raw_hit_pos, include_detail)]


def search_entries_chunk(
    entries: List[Dict],
    norm_keyword_groups: List[List[str]],
    raw_keywords: List[str],
    search_mode: str,
    range_limit: int,
    space_mode: str,
    normalize_mode: str,
    include_detail: bool,
) -> List[Dict]:
    chunk_results: List[Dict] = []
    for entry in entries:
        hit = search_text_logic(
            entry,
            norm_keyword_groups,
            raw_keywords,
            search_mode,
            range_limit,
            space_mode,
            normalize_mode,
            include_detail,
        )
        if hit:
            chunk_results.extend(hit)
    return chunk_results


def build_query_groups(query: str, space_mode: str, normalize_mode: str) -> Tuple[List[str], List[List[str]]]:
    keywords = [k for k in (query or "").split() if k.strip()]
    normalizer = normalize_text_nfkc_casefold if normalize_mode == "normalized" else normalize_text_minimal
    norm_keyword_groups = [
        [apply_space_mode(normalizer(k).strip(), space_mode)]
        for k in keywords
        if k.strip()
    ]
    return keywords, norm_keyword_groups


def check_range_match(norm_text: str, norm_keyword_groups: List[List[str]], range_limit: int) -> bool:
    positions = []
    for group_id, group in enumerate(norm_keyword_groups):
        for kw in group:
            if not kw:
                continue
            start = 0
            while True:
                idx = norm_text.find(kw, start)
                if idx == -1:
                    break
                positions.append((idx, group_id))
                start = idx + 1
    positions.sort()
    left = 0
    count_map = defaultdict(int)
    unique_count = 0
    target_unique = len(norm_keyword_groups)
    for right in range(len(positions)):
        r_idx, r_group_id = positions[right]
        if count_map[r_group_id] == 0:
            unique_count += 1
        count_map[r_group_id] += 1
        while unique_count == target_unique:
            l_idx, l_group_id = positions[left]
            current_len = r_idx - l_idx
            if current_len <= range_limit:
                return True
            count_map[l_group_id] -= 1
            if count_map[l_group_id] == 0:
                unique_count -= 1
            left += 1
    return False


def search_text_logic_shared(
    entry: Dict,
    mm: mmap.mmap,
    norm_keyword_groups: List[List[str]],
    norm_keyword_groups_bytes: List[List[bytes]],
    raw_keywords: List[str],
    search_mode: str,
    range_limit: int,
    space_mode: str,
    normalize_mode: str,
    include_detail: bool,
) -> List[Dict]:
    if not norm_keyword_groups_bytes:
        return []

    if normalize_mode == "normalized":
        if space_mode == "all":
            norm_start = entry.get("norm_cf_all_off", entry["norm_all_off"])
            norm_len = entry.get("norm_cf_all_len", entry["norm_all_len"])
        elif space_mode == "jp":
            norm_start = entry.get("norm_cf_jp_off", entry["norm_jp_off"])
            norm_len = entry.get("norm_cf_jp_len", entry["norm_jp_len"])
        else:
            norm_start = entry.get("norm_cf_off", entry["norm_off"])
            norm_len = entry.get("norm_cf_len", entry["norm_len"])
    else:
        if space_mode == "all":
            norm_start = entry.get("norm_strict_all_off", entry["norm_all_off"])
            norm_len = entry.get("norm_strict_all_len", entry["norm_all_len"])
        elif space_mode == "jp":
            norm_start = entry.get("norm_strict_jp_off", entry["norm_jp_off"])
            norm_len = entry.get("norm_strict_jp_len", entry["norm_jp_len"])
        else:
            norm_start = entry.get("norm_strict_off", entry["norm_off"])
            norm_len = entry.get("norm_strict_len", entry["norm_len"])
    norm_end = norm_start + norm_len

    is_match = False

    if search_mode == "OR":
        for group in norm_keyword_groups_bytes:
            for k in group:
                if not k:
                    continue
                if mm.find(k, norm_start, norm_end) != -1:
                    is_match = True
                    break
            if is_match:
                break
    elif search_mode == "AND":
        for group in norm_keyword_groups_bytes:
            if not any(mm.find(k, norm_start, norm_end) != -1 for k in group if k):
                return []
        if range_limit == 0 or len(norm_keyword_groups_bytes) == 1:
            is_match = True
        else:
            norm_text = mm[norm_start:norm_end].decode("utf-8", errors="ignore")
            is_match = check_range_match(norm_text, norm_keyword_groups, range_limit)
        if not is_match:
            return []

    if not is_match:
        return []

    raw_start = entry["raw_off"]
    raw_len = entry["raw_len"]
    raw_end = raw_start + raw_len
    raw_bytes = mm[raw_start:raw_end]
    raw_text = raw_bytes.decode("utf-8", errors="ignore")
    raw_for_positions = normalize_invisible_separators(raw_text)

    raw_hit_pos = _find_raw_hit_position(raw_for_positions, raw_keywords, space_mode)
    return [_build_search_result(entry, raw_text, raw_for_positions, raw_hit_pos, include_detail)]


def search_entries_chunk_shared(
    entries: List[Dict],
    mm: mmap.mmap,
    norm_keyword_groups: List[List[str]],
    norm_keyword_groups_bytes: List[List[bytes]],
    raw_keywords: List[str],
    search_mode: str,
    range_limit: int,
    space_mode: str,
    normalize_mode: str,
    include_detail: bool,
) -> List[Dict]:
    chunk_results: List[Dict] = []
    for entry in entries:
        hit = search_text_logic_shared(
            entry,
            mm,
            norm_keyword_groups,
            norm_keyword_groups_bytes,
            raw_keywords,
            search_mode,
            range_limit,
            space_mode,
            normalize_mode,
            include_detail,
        )
        if hit:
            chunk_results.extend(hit)
    return chunk_results


def build_memory_pages(folder_id: str, folder_name: str, cache: Dict[str, Dict]) -> List[Dict]:
    """Build memory pages from cache for search (worker process version).

    Note: This is used when SEARCH_PROCESS_SHARED=0. Unlike routes.py version,
    displayPath is omitted here; _build_search_result applies host_aliases.
    """
    from .utils import env_bool
    store_normalized = env_bool("INDEX_STORE_NORMALIZED", True)
    pages: List[Dict] = []
    for path, meta in cache.items():
        file_name = os.path.basename(path)
        is_pdf = file_name.lower().endswith(".pdf")
        ext = os.path.splitext(file_name)[1].lower()
        is_excel = ext in {".xlsx", ".xls"}
        file_id = file_id_from_path(path)
        data = meta.get("data")
        if not data:
            continue
        for page_num, raw_text in data.items():
            if not raw_text:
                continue
            # Use same normalization as routes.py
            norm_base = normalize_text(raw_text)
            norm_strict = normalize_text_minimal(raw_text)
            norm_cf_base = (
                normalize_text_nfkc_casefold(raw_text) if store_normalized else ""
            )
            # Use "-" for non-PDF/Excel (consistent with routes.py)
            page_display = "-" if not (is_pdf or is_excel) else page_num
            entry = {
                "folderId": folder_id,
                "folderName": folder_name,
                "file": file_name,
                "path": path,
                # displayPath is omitted; _build_search_result applies host_aliases
                "page": page_display,
                "pageRaw": page_num,
                "raw": raw_text,
                "norm": norm_base,
                "norm_jp": apply_space_mode(norm_base, "jp"),
                "norm_all": apply_space_mode(norm_base, "all"),
                "norm_strict": norm_strict,
                "norm_strict_jp": apply_space_mode(norm_strict, "jp"),
                "norm_strict_all": apply_space_mode(norm_strict, "all"),
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


def init_search_worker(folder_snapshot: List[Dict], aliases: Dict[str, str]):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global WORKER_MEMORY_PAGES, WORKER_FOLDER_META, host_aliases
    host_aliases = aliases
    pages_map: Dict[str, List[Dict]] = {}
    meta_map: Dict[str, Tuple[str, str]] = {}
    for folder in folder_snapshot:
        folder_id = folder["id"]
        folder_path = folder["path"]
        folder_name = folder["name"]
        cache = load_index_from_disk(folder_path)
        pages_map[folder_id] = build_memory_pages(folder_id, folder_name, cache)
        meta_map[folder_id] = (folder_path, folder_name)
    WORKER_MEMORY_PAGES = pages_map
    WORKER_FOLDER_META = meta_map


def init_search_worker_shared(folder_snapshot: List[Dict], aliases: Dict[str, str]):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global WORKER_SHARED_ENTRIES, WORKER_SHARED_MM, WORKER_SHARED_FILES, host_aliases, PROCESS_SHARED
    host_aliases = aliases
    PROCESS_SHARED = True
    WORKER_SHARED_ENTRIES = {}
    WORKER_SHARED_MM = {}
    WORKER_SHARED_FILES = {}
    for folder in folder_snapshot:
        folder_id = folder["id"]
        shared_path = folder["shared_path"]
        entries = folder["entries"]
        if not entries:
            WORKER_SHARED_ENTRIES[folder_id] = []
            continue
        try:
            fh = open(shared_path, "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            log_info(f"共有ストア読込失敗: {shared_path}")
            continue
        WORKER_SHARED_FILES[folder_id] = fh
        WORKER_SHARED_MM[folder_id] = mm
        WORKER_SHARED_ENTRIES[folder_id] = entries


def perform_search_process(
    target_ids: List[str],
    params,
    norm_keyword_groups: List[List[str]],
    raw_keywords: List[str],
    worker_count: int,
    include_detail: bool,
) -> List[Dict]:
    if PROCESS_SHARED:
        return perform_search_process_shared(
            target_ids,
            params,
            norm_keyword_groups,
            raw_keywords,
            worker_count,
            include_detail,
        )

    results: List[Dict] = []

    def search_folder(fid: str) -> List[Dict]:
        entries = WORKER_MEMORY_PAGES.get(fid)
        if not entries:
            meta = WORKER_FOLDER_META.get(fid)
            if meta:
                folder_path, folder_name = meta
                cache = load_index_from_disk(folder_path)
                entries = build_memory_pages(fid, folder_name, cache)
                WORKER_MEMORY_PAGES[fid] = entries
            else:
                entries = []
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
        return folder_results

    for fid in target_ids:
        try:
            res = search_folder(fid)
            if res:
                results.extend(res)
        except Exception:
            continue
    return results


def perform_search_process_shared(
    target_ids: List[str],
    params,
    norm_keyword_groups: List[List[str]],
    raw_keywords: List[str],
    worker_count: int,
    include_detail: bool,
) -> List[Dict]:
    results: List[Dict] = []
    norm_keyword_groups_bytes = [
        [encode_norm_text(k) for k in group if k] for group in norm_keyword_groups
    ]

    def search_folder(fid: str) -> List[Dict]:
        entries = WORKER_SHARED_ENTRIES.get(fid, [])
        mm = WORKER_SHARED_MM.get(fid)
        if not mm or not entries:
            return []
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
                        search_entries_chunk_shared,
                        chunk,
                        mm,
                        norm_keyword_groups,
                        norm_keyword_groups_bytes,
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
            folder_results = search_entries_chunk_shared(
                entries_iter,
                mm,
                norm_keyword_groups,
                norm_keyword_groups_bytes,
                raw_keywords,
                params.mode,
                params.range_limit,
                params.space_mode,
                params.normalize_mode,
                include_detail,
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
