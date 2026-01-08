import asyncio
import csv
import gzip
import hashlib
import json
import mmap
import os
import pickle
import re
import socket
import subprocess
import threading
import time
import unicodedata
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

SYSTEM_VERSION = "1.1.0"
# File Version: 1.2.0
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
try:
    from colorama import Fore, Style, init as colorama_init
except Exception:
    Fore = None
    Style = None
    colorama_init = None

# --- 外部依存ロジック ---
from docx import Document
from openpyxl import load_workbook
from pdfminer.high_level import extract_pages, extract_text as miner_extract_text
from pdfminer.layout import LAParams, LTTextBoxHorizontal, LTTextBoxVertical, LTTextContainer
from pdfminer.pdfpage import PDFPage

# --- 定数 ---
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "indexes"
INDEX_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"}
INDEX_VERSION = "v3"
SEARCH_CACHE_VERSION = "v1"
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
FIXED_CACHE_INDEX = CACHE_DIR / "fixed_cache_index.json"
QUERY_STATS_PATH = CACHE_DIR / "query_stats.json"

# --- 世代ディレクトリ定数 ---
BUILD_DIR = INDEX_DIR / ".build"
BUILD_DIR.mkdir(exist_ok=True)
CURRENT_POINTER_FILE = INDEX_DIR / "current.txt"

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


# --- ユーティリティ ---
CID_PATTERN = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
INVISIBLE_SEPARATORS_PATTERN = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00a0\u202f]")
JP_CHAR_CLASS = r"\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uFF66-\uFF9F"
OCR_PAGE_PATTERN = re.compile(r"^\s*-{2,}\s*Page\s*(\d+)?\s*-{2,}\s*$", re.IGNORECASE)


def strip_cid(text: str) -> str:
    if not text:
        return ""
    return CID_PATTERN.sub("", text)


def normalize_invisible_separators(text: str) -> str:
    if not text:
        return ""
    return INVISIBLE_SEPARATORS_PATTERN.sub(" ", text)


def build_flexible_keyword_regex(keyword: str) -> re.Pattern | None:
    compact = re.sub(r"\s+", "", keyword or "")
    if len(compact) < 2 or len(compact) > 64:
        return None
    parts = [re.escape(ch) for ch in compact]
    pattern = r"\s*".join(parts)
    return re.compile(pattern, re.IGNORECASE)


def apply_space_mode(text: str, mode: str) -> str:
    if not text:
        return ""
    if mode == "all":
        return re.sub(r"\s+", "", text)
    if mode == "jp":
        pattern = rf"(?<=[{JP_CHAR_CLASS}])\s+(?=[{JP_CHAR_CLASS}])"
        return re.sub(pattern, "", text)
    return text


def normalize_text(text: str) -> str:
    """キーワード検索のための正規化."""
    if not text:
        return ""
    text = strip_cid(text)
    text = unicodedata.normalize("NFKC", text)
    text = normalize_invisible_separators(text)
    text = text.lower()
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def encode_norm_text(text: str) -> bytes:
    if not text:
        return b""
    return text.encode("utf-8")


def normalize_snippet_text(text: str) -> str:
    text = strip_cid(text)
    text = normalize_invisible_separators(text)
    text = text.replace("\r", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_detail_text(text: str) -> str:
    text = strip_cid(text)
    text = normalize_invisible_separators(text)
    text = text.replace("\r", "")
    text = re.sub(r"\n(?!　)", " ", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


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
            for match in re.finditer(r"\\binet\\s+(\\d+\\.\\d+\\.\\d+\\.\\d+)", output):
                ip = match.group(1)
                if ip and not ip.startswith("127."):
                    ips.add(ip)
        except Exception:
            pass
    return sorted(ips)


def split_by_ocr_markers(text: str) -> Dict[int, str]:
    lines = text.split("\n")
    pages: Dict[int, str] = {}
    buffer: List[str] = []
    current_page = 1
    found_marker = False
    for line in lines:
        match = OCR_PAGE_PATTERN.match(line)
        if match:
            found_marker = True
            if buffer:
                pages[current_page] = "\n".join(buffer).strip()
                buffer = []
            page_hint = match.group(1)
            if page_hint and page_hint.isdigit():
                current_page = int(page_hint)
            else:
                current_page += 1
            continue
        buffer.append(line)
    if buffer:
        pages[current_page] = "\n".join(buffer).strip()
    if not found_marker:
        return {}
    return {k: v for k, v in pages.items() if v}


def find_break_pos(text: str, start: int, target: int, min_size: int, max_size: int) -> int:
    end_limit = min(len(text), start + max_size)
    target_pos = min(len(text), start + target)
    window_start = min(len(text), start + min_size)
    if window_start >= end_limit:
        return end_limit

    primary = set(["\n", " ", "\t"])
    secondary = set("。．、，.!?？！)]}」』】）〕〉》】○×△□◇■-―—")

    def scan_for(chars: set) -> int | None:
        for pos in range(target_pos, end_limit):
            if text[pos] in chars:
                return pos
        for pos in range(target_pos - 1, window_start - 1, -1):
            if text[pos] in chars:
                return pos
        return None

    hit = scan_for(primary)
    if hit is None:
        hit = scan_for(secondary)
    if hit is None:
        return min(start + target, len(text))
    while hit + 1 < len(text) and text[hit + 1] in primary:
        hit += 1
    return hit + 1


def split_text_by_size(
    text: str,
    target: int = 1200,
    min_size: int = 600,
    max_size: int = 1800,
    overlap: int = 200,
) -> Dict[int, str]:
    pages: Dict[int, str] = {}
    pos = 0
    page_num = 1
    length = len(text)
    if length <= max_size:
        cleaned = text.strip()
        return {1: cleaned} if cleaned else {}
    while pos < length:
        remaining = length - pos
        if remaining <= max_size:
            chunk = text[pos:].strip()
            if chunk:
                pages[page_num] = chunk
            break
        split_pos = find_break_pos(text, pos, target, min_size, max_size)
        if split_pos <= pos:
            split_pos = min(pos + target, length)
        if split_pos <= pos:
            break
        chunk = text[pos:split_pos].strip()
        if chunk:
            pages[page_num] = chunk
            page_num += 1
        if overlap > 0:
            pos = max(split_pos - overlap, pos + 1)
        else:
            pos = split_pos
    return pages


def split_non_pdf_text(text: str) -> Dict[int, str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    pages = split_by_ocr_markers(normalized)
    if pages:
        return pages
    return split_text_by_size(normalized)


def read_text_file_safe(path: str) -> str:
    encodings = ["utf-8", "cp932", "shift_jis", "euc-jp"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""


def read_csv_file_safe(path: str) -> str:
    encodings = ["utf-8", "cp932", "shift_jis"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.reader(f)
                return "\n".join([" ".join(row) for row in reader])
        except Exception:
            continue
    return ""


def read_excel_rows_xlsx(path: str) -> List[Tuple[str, int, str]]:
    try:
        wb = load_workbook(path, read_only=True, data_only=True)
    except Exception:
        return []

    rows: List[Tuple[str, int, str]] = []
    try:
        for ws in wb.worksheets:
            sheet_name = ws.title or "Sheet"
            row_idx = 0
            for row in ws.iter_rows(values_only=True):
                row_idx += 1
                cells = [
                    str(cell).strip()
                    for cell in row
                    if cell is not None and str(cell).strip()
                ]
                if cells:
                    rows.append((sheet_name, row_idx, " ".join(cells)))
    except Exception:
        return []
    finally:
        try:
            wb.close()
        except Exception:
            pass
    return rows


def read_excel_rows_xls(path: str) -> List[Tuple[str, int, str]]:
    try:
        import xlrd
    except Exception:
        return []
    try:
        wb = xlrd.open_workbook(path)
    except Exception:
        return []

    rows: List[Tuple[str, int, str]] = []
    try:
        for sheet in wb.sheets():
            sheet_name = sheet.name or "Sheet"
            for r in range(sheet.nrows):
                row_values = [
                    str(sheet.cell_value(r, c)).strip()
                    for c in range(sheet.ncols)
                    if str(sheet.cell_value(r, c)).strip()
                ]
                if row_values:
                    rows.append((sheet_name, r + 1, " ".join(row_values)))
    except Exception:
        return []
    return rows


def split_excel_rows(rows: List[Tuple[str, int, str]], rows_per_page: int = 40) -> Dict[str, str]:
    pages: Dict[str, str] = {}
    by_sheet: Dict[str, List[Tuple[int, str]]] = {}
    for sheet, row_idx, text in rows:
        by_sheet.setdefault(sheet, []).append((row_idx, text))

    for sheet, sheet_rows in by_sheet.items():
        sheet_rows.sort(key=lambda x: x[0])
        for i in range(0, len(sheet_rows), rows_per_page):
            chunk = sheet_rows[i : i + rows_per_page]
            start_row = chunk[0][0]
            end_row = chunk[-1][0]
            label = f"{sheet} 行 {start_row}-{end_row}"
            pages[label] = "\n".join(text for _, text in chunk).strip()
    return pages


def iter_text_containers(obj):
    if isinstance(obj, LTTextContainer):
        yield obj
        return
    for child in getattr(obj, "_objs", []):
        yield from iter_text_containers(child)


def order_text_boxes(boxes: List[Dict], page_bbox: Tuple[float, float, float, float]) -> str:
    if not boxes:
        return ""
    x0, y0, x1, y1 = page_bbox
    page_mid = (x0 + x1) / 2.0
    vertical_count = sum(1 for b in boxes if b["vertical"])
    horizontal_count = len(boxes) - vertical_count
    use_vertical = vertical_count > horizontal_count and vertical_count >= 3

    if use_vertical:
        ordered = sorted(boxes, key=lambda b: (-b["x0"], -b["y1"]))
    else:
        left = [b for b in boxes if b["cx"] < page_mid]
        right = [b for b in boxes if b["cx"] >= page_mid]
        is_two_column = (
            len(boxes) >= 6
            and len(left) >= 2
            and len(right) >= 2
            and 0.3 < (len(left) / len(boxes)) < 0.7
        )
        if is_two_column:
            left.sort(key=lambda b: (-b["y1"], b["x0"]))
            right.sort(key=lambda b: (-b["y1"], b["x0"]))
            ordered = left + right
        else:
            ordered = sorted(boxes, key=lambda b: (-b["y1"], b["x0"]))

    return "\n".join(b["text"].strip() for b in ordered if b["text"].strip())


def extract_pdf_text_by_layout(file_path: str) -> Dict[int, str]:
    page_texts: Dict[int, str] = {}
    laparams = LAParams()
    for page_num, layout in enumerate(extract_pages(file_path, laparams=laparams), start=1):
        boxes = []
        for container in iter_text_containers(layout):
            text = container.get_text()
            if not text or not text.strip():
                continue
            x0, y0, x1, y1 = container.bbox
            boxes.append(
                {
                    "text": text,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "cx": (x0 + x1) / 2.0,
                    "vertical": isinstance(container, LTTextBoxVertical),
                }
            )
        if boxes:
            ordered_text = order_text_boxes(boxes, layout.bbox)
            if ordered_text:
                page_texts[page_num] = ordered_text
    return page_texts


def extract_text_from_file(file_path: str) -> Dict[int, str]:
    """各ファイルからテキストを抽出（PDFは高精度モード固定）."""
    texts, _ = extract_text_from_file_with_reason(file_path)
    return texts


def empty_reason(file_path: str, fallback: str) -> str:
    try:
        if os.path.getsize(file_path) == 0:
            return "空ファイル"
    except Exception:
        return fallback
    return fallback


def extract_text_from_file_with_reason(file_path: str) -> Tuple[Dict[int, str], str]:
    """抽出結果と理由を返す."""
    path_obj = Path(file_path)
    ext = path_obj.suffix.lower()
    page_texts: Dict[int, str] = {}
    file_path_str = str(file_path)
    reason = ""

    try:
        if ext == ".pdf":
            try:
                page_texts = extract_pdf_text_by_layout(file_path_str)
            except Exception:
                page_texts = {}
            if not page_texts:
                laparams = LAParams()
                with open(file_path_str, "rb") as fp:
                    pages = list(PDFPage.get_pages(fp, check_extractable=False))
                    total_pages = len(pages)

                with open(file_path_str, "rb") as fp:
                    for i in range(total_pages):
                        try:
                            text = miner_extract_text(fp, page_numbers=[i], laparams=laparams)
                            if text and text.strip():
                                page_texts[i + 1] = text
                        except Exception:
                            continue
            if not page_texts:
                reason = empty_reason(file_path_str, "PDF抽出失敗")

        elif ext == ".docx":
            doc = Document(file_path_str)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        full_text.append(cell.text)
            content = "\n".join(full_text)
            if content:
                page_texts = split_non_pdf_text(content)
            else:
                reason = empty_reason(file_path_str, "内容が空")

        elif ext == ".txt":
            content = read_text_file_safe(file_path_str)
            if content:
                page_texts = split_non_pdf_text(content)
            else:
                reason = empty_reason(file_path_str, "読み取り失敗または内容が空")

        elif ext == ".csv":
            content = read_csv_file_safe(file_path_str)
            if content:
                page_texts = split_non_pdf_text(content)
            else:
                reason = empty_reason(file_path_str, "読み取り失敗または内容が空")
        elif ext == ".xlsx":
            rows = read_excel_rows_xlsx(file_path_str)
            if rows:
                page_texts = split_excel_rows(rows)
            else:
                reason = empty_reason(file_path_str, "読み取り失敗または内容が空")
        elif ext == ".xls":
            rows = read_excel_rows_xls(file_path_str)
            if rows:
                page_texts = split_excel_rows(rows)
            else:
                reason = empty_reason(file_path_str, "読み取り失敗または内容が空")
    except Exception as exc:
        return {}, f"例外:{type(exc).__name__}"

    if page_texts:
        return page_texts, ""
    return {}, reason or "抽出結果が空"


def _find_raw_hit_position(
    raw_for_positions: str,
    raw_keywords: List[str],
    space_mode: str,
) -> int:
    """生テキスト内でキーワードの位置を検索する共通ヘルパー."""
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
) -> Dict:
    """検索結果のスニペットと詳細を生成する共通ヘルパー."""
    snippet_start = max(0, raw_hit_pos - SNIPPET_PREFIX_CHARS) if raw_hit_pos != -1 else 0
    snippet_end = min(len(raw_for_positions), snippet_start + SNIPPET_TOTAL_LENGTH)
    snippet_raw = raw_for_positions[snippet_start:snippet_end]
    snippet = f"...{normalize_snippet_text(snippet_raw)}..."

    detail_pos = raw_hit_pos if raw_hit_pos != -1 else 0
    detail_start = max(0, detail_pos - DETAIL_CONTEXT_PREFIX)
    detail_end = min(len(raw_text), detail_start + DETAIL_WINDOW_SIZE)
    detail_text_raw = raw_text[detail_start:detail_end]
    detail_text = normalize_detail_text(detail_text_raw)

    return {
        "file": entry["file"],
        "path": entry["path"],
        "displayPath": entry.get(
            "displayPath", display_path_for_path(entry["path"], host_aliases)
        ),
        "page": entry["page"],
        "context": snippet,
        "detail": detail_text,
        "folderId": entry["folderId"],
        "folderName": entry["folderName"],
    }


def search_text_logic(
    entry: Dict,
    norm_keyword_groups: List[List[str]],
    raw_keywords: List[str],
    search_mode: str,
    range_limit: int,
    space_mode: str,
) -> List[Dict]:
    if not norm_keyword_groups:
        return []

    raw_text = entry["raw"]
    raw_for_positions = normalize_invisible_separators(raw_text)
    if space_mode == "all":
        norm_text = entry["norm_all"]
    elif space_mode == "jp":
        norm_text = entry["norm_jp"]
    else:
        norm_text = entry["norm"]
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
    return [_build_search_result(entry, raw_text, raw_for_positions, raw_hit_pos)]


def search_entries_chunk(
    entries: List[Dict],
    norm_keyword_groups: List[List[str]],
    raw_keywords: List[str],
    search_mode: str,
    range_limit: int,
    space_mode: str,
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
        )
        if hit:
            chunk_results.extend(hit)
    return chunk_results


def build_query_groups(query: str, space_mode: str) -> Tuple[List[str], List[List[str]]]:
    keywords = [k for k in (query or "").split() if k.strip()]
    norm_keyword_groups = [
        [apply_space_mode(normalize_text(k).strip(), space_mode)]
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
) -> List[Dict]:
    if not norm_keyword_groups_bytes:
        return []

    if space_mode == "all":
        norm_start = entry["norm_all_off"]
        norm_len = entry["norm_all_len"]
    elif space_mode == "jp":
        norm_start = entry["norm_jp_off"]
        norm_len = entry["norm_jp_len"]
    else:
        norm_start = entry["norm_off"]
        norm_len = entry["norm_len"]
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
    return [_build_search_result(entry, raw_text, raw_for_positions, raw_hit_pos)]


def search_entries_chunk_shared(
    entries: List[Dict],
    mm: mmap.mmap,
    norm_keyword_groups: List[List[str]],
    norm_keyword_groups_bytes: List[List[bytes]],
    raw_keywords: List[str],
    search_mode: str,
    range_limit: int,
    space_mode: str,
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
        )
        if hit:
            chunk_results.extend(hit)
    return chunk_results


def perform_search(
    target_ids: List[str],
    params: "SearchParams",
    norm_keyword_groups: List[List[str]],
    raw_keywords: List[str],
    worker_count: int,
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


def init_search_worker(folder_snapshot: List[Dict], aliases: Dict[str, str]):
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
    params: "SearchParams",
    norm_keyword_groups: List[List[str]],
    raw_keywords: List[str],
    worker_count: int,
) -> List[Dict]:
    if PROCESS_SHARED:
        return perform_search_process_shared(
            target_ids,
            params,
            norm_keyword_groups,
            raw_keywords,
            worker_count,
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
    params: "SearchParams",
    norm_keyword_groups: List[List[str]],
    raw_keywords: List[str],
    worker_count: int,
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
) -> Tuple[List[Dict], List[str]]:
    raw_keywords, norm_keyword_groups = build_query_groups(query, params.space_mode)
    worker_count = search_worker_count
    if search_execution_mode == "process" and search_executor:
        future = search_executor.submit(
            perform_search_process,
            target_ids,
            params,
            norm_keyword_groups,
            raw_keywords,
            worker_count,
        )
        results = future.result()
    else:
        results = perform_search(
            target_ids,
            params,
            norm_keyword_groups,
            raw_keywords,
            worker_count,
        )
    return results, raw_keywords


# --- インデックス関連 ---

# --- 世代管理ユーティリティ ---
def create_generation_uuid() -> str:
    """新しい世代UUIDを生成（タイムスタンプ + ランダム）."""
    import uuid
    timestamp = int(time.time())
    random_part = uuid.uuid4().hex[:8]
    return f"{timestamp}_{random_part}"


def get_current_generation_pointer() -> str | None:
    """currentポインターファイルから世代名を取得."""
    if not CURRENT_POINTER_FILE.exists():
        return None
    try:
        gen_name = CURRENT_POINTER_FILE.read_text(encoding="utf-8").strip()
        return gen_name if gen_name else None
    except Exception:
        return None


def set_current_generation_pointer(gen_name: str) -> None:
    """currentポインターファイルに世代名を設定（原子的）."""
    temp_file = CURRENT_POINTER_FILE.with_suffix(".tmp")
    try:
        temp_file.write_text(gen_name, encoding="utf-8")
        temp_file.replace(CURRENT_POINTER_FILE)
    except Exception as e:
        log_warn(f"currentポインター設定失敗: {e}")
        if temp_file.exists():
            temp_file.unlink()


def get_generation_dir(gen_name: str | None = None, build: bool = False) -> Path:
    """世代ディレクトリのパスを取得."""
    if gen_name is None:
        gen_name = get_current_generation_pointer()
        if gen_name is None:
            return INDEX_DIR
    base = BUILD_DIR if build else INDEX_DIR
    return base / f"gen_{gen_name}"


def get_current_generation_dir() -> Path:
    """現在の世代ディレクトリを取得（存在しない場合は INDEX_DIR を返す）."""
    gen_name = get_current_generation_pointer()
    if gen_name:
        gen_dir = get_generation_dir(gen_name, build=False)
        if gen_dir.exists():
            return gen_dir
    return INDEX_DIR


def create_manifest(gen_name: str, gen_dir: Path, folder_states_snapshot: Dict) -> Dict:
    """manifest.json を作成."""
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)

    folder_paths = sorted([meta["path"] for meta in folder_states_snapshot.values()])
    source_folders_hash = hashlib.sha256(
        "|".join(folder_paths).encode("utf-8")
    ).hexdigest()

    folders_info = {}
    for folder_id, meta in folder_states_snapshot.items():
        path = meta["path"]
        hashed = hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
        index_file = f"index_{hashed}_{INDEX_VERSION}.pkl.gz"
        index_path = gen_dir / index_file
        file_count = 0
        if index_path.exists():
            try:
                with gzip.open(index_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        file_count = len(data)
            except Exception:
                pass

        folders_info[folder_id] = {
            "path": path,
            "name": meta["name"],
            "file_count": file_count,
            "index_file": index_file,
        }

    manifest = {
        "index_uuid": gen_name,
        "schema_version": INDEX_VERSION,
        "created_at": now.isoformat(),
        "created_timestamp": int(now.timestamp()),
        "source_folders_hash": source_folders_hash,
        "folders": folders_info,
    }

    return manifest


def save_manifest(gen_dir: Path, manifest: Dict) -> None:
    """manifest.json を保存."""
    manifest_path = gen_dir / "manifest.json"
    try:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_warn(f"manifest.json 保存失敗: {e}")


def load_manifest(gen_dir: Path) -> Dict | None:
    """manifest.json を読み込み."""
    manifest_path = gen_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_generations() -> List[Tuple[str, Path, Dict | None]]:
    """すべての世代をリストアップ（世代名, パス, manifest）."""
    generations = []
    for item in INDEX_DIR.iterdir():
        if item.is_dir() and item.name.startswith("gen_"):
            gen_name = item.name[4:]  # "gen_" を除く
            manifest = load_manifest(item)
            generations.append((gen_name, item, manifest))

    # タイムスタンプでソート（新しい順）
    generations.sort(key=lambda x: x[2].get("created_timestamp", 0) if x[2] else 0, reverse=True)
    return generations


def cleanup_old_generations(current_gen_name: str | None, grace_sec: int = 300) -> None:
    """古い世代を削除（保持ポリシーに従う）."""
    import shutil

    keep_generations = env_int("INDEX_KEEP_GENERATIONS", 3)
    keep_days = env_int("INDEX_KEEP_DAYS", 0)
    max_bytes = env_int("INDEX_MAX_BYTES", 0)

    generations = list_generations()

    # 現在の世代を特定
    current_gen_dir = None
    if current_gen_name:
        current_gen_dir = get_generation_dir(current_gen_name, build=False)

    # 削除候補リスト
    to_delete = []
    total_bytes = 0

    for i, (gen_name, gen_dir, manifest) in enumerate(generations):
        # 現在の世代は絶対に削除しない
        if current_gen_dir and gen_dir == current_gen_dir:
            continue

        # 猶予期間内は削除しない
        if manifest and grace_sec > 0:
            created_timestamp = manifest.get("created_timestamp", 0)
            age_sec = time.time() - created_timestamp
            if age_sec < grace_sec:
                continue

        # 世代数チェック（current を除外した上で保持数を確認）
        # current を除いた index を計算
        non_current_index = i
        if current_gen_dir:
            # current より前の世代の数を数える
            non_current_index = sum(1 for j in range(i) if generations[j][1] != current_gen_dir)

        if keep_generations > 0 and non_current_index >= keep_generations:
            to_delete.append((gen_name, gen_dir, "世代数超過"))
            continue

        # 日数チェック
        if manifest and keep_days > 0:
            created_timestamp = manifest.get("created_timestamp", 0)
            age_days = (time.time() - created_timestamp) / 86400
            if age_days > keep_days:
                to_delete.append((gen_name, gen_dir, "保持期限切れ"))
                continue

        # 容量計算
        if max_bytes > 0:
            try:
                dir_bytes = sum(
                    f.stat().st_size for f in gen_dir.rglob("*") if f.is_file()
                )
                total_bytes += dir_bytes
            except Exception:
                pass

    # 容量超過チェック（古い順に削除）
    if max_bytes > 0 and total_bytes > max_bytes:
        for gen_name, gen_dir, manifest in reversed(generations):
            if current_gen_dir and gen_dir == current_gen_dir:
                continue
            if (gen_name, gen_dir, None) not in [(d[0], d[1], None) for d in to_delete]:
                try:
                    dir_bytes = sum(
                        f.stat().st_size for f in gen_dir.rglob("*") if f.is_file()
                    )
                    total_bytes -= dir_bytes
                    to_delete.append((gen_name, gen_dir, "容量超過"))
                    if total_bytes <= max_bytes:
                        break
                except Exception:
                    pass

    # 削除実行
    for gen_name, gen_dir, reason in to_delete:
        try:
            shutil.rmtree(gen_dir)
            log_info(f"世代削除: gen_{gen_name} 理由={reason}")
        except Exception as e:
            log_warn(f"世代削除失敗: gen_{gen_name} エラー={e}")


def index_path_for(folder_path: str, gen_dir: Path | None = None) -> Path:
    """インデックスファイルのパスを取得（世代ディレクトリ対応）."""
    hashed = hashlib.sha256(folder_path.encode("utf-8")).hexdigest()[:12]
    if gen_dir is None:
        gen_dir = get_current_generation_dir()
    return gen_dir / f"index_{hashed}_{INDEX_VERSION}.pkl.gz"


def load_index_from_disk(folder_path: str, gen_dir: Path | None = None) -> Dict[str, Dict]:
    """インデックスをディスクから読み込み（世代ディレクトリ対応）."""
    idx_path = index_path_for(folder_path, gen_dir)
    if not idx_path.exists():
        return {}
    try:
        with gzip.open(idx_path, "rb") as f:
            data = pickle.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_index_to_disk(folder_path: str, cache: Dict[str, Dict], gen_dir: Path | None = None):
    """インデックスをディスクに保存（世代ディレクトリ対応）."""
    idx_path = index_path_for(folder_path, gen_dir)
    # gen_dir が指定されている場合、ディレクトリを作成
    if gen_dir is not None:
        gen_dir.mkdir(parents=True, exist_ok=True)
    try:
        with gzip.open(idx_path, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        # 保存失敗時は静かにスキップ
        return


def scan_files(folder: str) -> List[Path]:
    root = Path(folder)
    files = []
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTS:
            files.append(f)
    return files


def build_index_for_folder(folder: str, previous_failures: Dict[str, str] | None = None, gen_dir: Path | None = None) -> Tuple[Dict[str, Dict], Dict, Dict[str, str]]:
    """インデックス作成（差分更新 & 高精度固定）."""
    start_time = time.time()
    existing_cache = load_index_from_disk(folder, gen_dir)
    failures = {k: v for k, v in (previous_failures or {}).items()}
    all_files = scan_files(folder)
    if not all_files and existing_cache:
        stats = {
            "total_files": len(existing_cache),
            "indexed_files": len(existing_cache),
            "updated_files": 0,
            "skipped_files": len(existing_cache),
        }
        elapsed = time.time() - start_time
        log_info(
            f"インデックス構築スキップ: {folder} 件数={stats['indexed_files']} "
            f"理由=ファイル一覧取得失敗 時間={elapsed:.1f}s"
        )
        failures = {k: v for k, v in failures.items() if k in existing_cache}
        return existing_cache, stats, failures
    if existing_cache and all_files:
        allow_shrink = os.getenv("REBUILD_ALLOW_SHRINK", "").strip().lower() in {"1", "true", "yes"}
        if not allow_shrink and len(all_files) < max(10, int(len(existing_cache) * 0.8)):
            stats = {
                "total_files": len(existing_cache),
                "indexed_files": len(existing_cache),
                "updated_files": 0,
                "skipped_files": len(existing_cache),
            }
            elapsed = time.time() - start_time
            log_info(
                f"インデックス構築スキップ: {folder} 件数={stats['indexed_files']} "
                f"理由=ファイル数急減 時間={elapsed:.1f}s"
            )
            failures = {k: v for k, v in failures.items() if k in existing_cache}
            return existing_cache, stats, failures

    current_map = {str(f): f for f in all_files}
    failures = {k: v for k, v in failures.items() if k in current_map}

    valid_cache = {
        path: meta for path, meta in existing_cache.items() if path in current_map
    }
    targets = []
    for path_str, path_obj in current_map.items():
        mtime = os.path.getmtime(path_str)
        cached = valid_cache.get(path_str)
        if cached and cached.get("mtime") == mtime and cached.get("high_acc"):
            continue
        targets.append(path_obj)

    updated_data: Dict[str, Dict] = {}
    if targets:
        max_workers = max(1, os.cpu_count() or 4)
        log_info(f"インデックス更新開始: {folder} 対象={len(targets)} workers={max_workers}")
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_path = {
                pool.submit(extract_text_from_file_with_reason, str(p)): str(p) for p in targets
            }
            for future in future_to_path:
                path_str = future_to_path[future]
                try:
                    texts, reason = future.result()
                    if texts:
                        updated_data[path_str] = {
                            "mtime": os.path.getmtime(path_str),
                            "data": texts,
                            "high_acc": True,
                        }
                        if path_str in failures:
                            failures.pop(path_str, None)
                    else:
                        failures[path_str] = reason or "抽出結果が空"
                        log_warn(f"インデックス失敗: {path_str} 理由={failures[path_str]}")
                except Exception:
                    failures[path_str] = "例外"
                    log_warn(f"インデックス失敗: {path_str} 理由=例外")
                    continue
        log_info(f"インデックス更新完了: {folder} 更新={len(updated_data)}")

    valid_cache.update(updated_data)
    failures = {k: v for k, v in failures.items() if k not in valid_cache}
    save_index_to_disk(folder, valid_cache, gen_dir)

    stats = {
        "total_files": len(all_files),
        "indexed_files": len(valid_cache),
        "updated_files": len(updated_data),
        "skipped_files": len(valid_cache) - len(updated_data),
    }
    elapsed = time.time() - start_time
    log_info(f"インデックス構築完了: {folder} 件数={stats['indexed_files']} 時間={elapsed:.1f}s")
    return valid_cache, stats, failures


def build_memory_pages(folder_id: str, folder_name: str, cache: Dict[str, Dict]) -> List[Dict]:
    pages: List[Dict] = []
    for path, meta in cache.items():
        file_name = os.path.basename(path)
        is_pdf = file_name.lower().endswith(".pdf")
        ext = os.path.splitext(file_name)[1].lower()
        is_excel = ext in {".xlsx", ".xls"}
        display_path = display_path_for_path(path, host_aliases)
        for page_num, raw_text in meta.get("data", {}).items():
            if not raw_text:
                continue
            norm_base = normalize_text(raw_text)
            page_display = "-" if not (is_pdf or is_excel) else page_num
            pages.append(
                {
                    "file": file_name,
                    "path": path,
                    "displayPath": display_path,
                    "page": page_display,
                    "raw": raw_text,
                    "norm": norm_base,
                    "norm_jp": apply_space_mode(norm_base, "jp"),
                    "norm_all": apply_space_mode(norm_base, "all"),
                    "folderId": folder_id,
                    "folderName": folder_name,
                }
            )
    return pages


def build_shared_blob(shared_path: Path, entries: List[Dict]) -> List[Dict]:
    offsets: List[Dict] = []
    total_size = 0
    for entry in entries:
        raw_b = entry["raw"].encode("utf-8")
        norm_b = encode_norm_text(entry["norm"])
        norm_jp_b = encode_norm_text(entry["norm_jp"])
        norm_all_b = encode_norm_text(entry["norm_all"])
        offsets.append(
            {
                "file": entry["file"],
                "path": entry["path"],
                "displayPath": entry["displayPath"],
                "page": entry["page"],
                "folderId": entry["folderId"],
                "folderName": entry["folderName"],
                "raw_off": total_size,
                "raw_len": len(raw_b),
                "norm_off": total_size + len(raw_b),
                "norm_len": len(norm_b),
                "norm_jp_off": total_size + len(raw_b) + len(norm_b),
                "norm_jp_len": len(norm_jp_b),
                "norm_all_off": total_size + len(raw_b) + len(norm_b) + len(norm_jp_b),
                "norm_all_len": len(norm_all_b),
            }
        )
        total_size += len(raw_b) + len(norm_b) + len(norm_jp_b) + len(norm_all_b)

    with open(shared_path, "wb") as f:
        f.truncate(total_size)
        for entry in entries:
            f.write(entry["raw"].encode("utf-8"))
            f.write(encode_norm_text(entry["norm"]))
            f.write(encode_norm_text(entry["norm_jp"]))
            f.write(encode_norm_text(entry["norm_all"]))
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


# --- グローバル状態 ---
app = FastAPI(title="Preloaded Folder Search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

load_dotenv()


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


host_aliases = parse_host_aliases()
configured_folders = parse_configured_folders()
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

WORKER_MEMORY_PAGES: Dict[str, List[Dict]] = {}
WORKER_FOLDER_META: Dict[str, Tuple[str, str]] = {}


@dataclass(frozen=True)
class SearchParams:
    mode: str
    range_limit: int
    space_mode: str


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


def folder_id_from_path(path: str) -> str:
    return hashlib.sha256(path.encode("utf-8")).hexdigest()[:10]


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


def build_all_indexes():
    """インデックス構築（世代ディレクトリ方式）."""
    import shutil

    log_info("インデックス構築開始")

    # 新しい世代を作成
    gen_name = create_generation_uuid()
    build_gen_dir = get_generation_dir(gen_name, build=True)
    build_gen_dir.mkdir(parents=True, exist_ok=True)

    log_info(f"世代ディレクトリ構築: gen_{gen_name}")

    # 各フォルダのインデックスを構築
    temp_indexes = {}
    temp_failures = {}

    for folder_id, meta in folder_states.items():
        path = meta["path"]
        if not os.path.isdir(path):
            meta["ready"] = False
            meta["message"] = f"フォルダが見つかりません: {path}"
            continue

        meta["message"] = "インデックス構築中..."
        log_info(f"フォルダ処理開始: {path}")

        # 既存の世代から前回のインデックスを読み込み（差分更新用）
        prev_gen_dir = get_current_generation_dir()
        prev_cache = load_index_from_disk(path, prev_gen_dir)

        prev_failures = index_failures.get(folder_id, {})
        cache, stats, failures = build_index_for_folder(path, prev_failures, build_gen_dir)

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
    try:
        build_gen_dir.replace(final_gen_dir)
        log_info(f"世代ディレクトリ移動完了: gen_{gen_name}")
    except Exception as e:
        log_warn(f"世代ディレクトリ移動失敗: {e}")
        # ビルドディレクトリを削除
        if build_gen_dir.exists():
            shutil.rmtree(build_gen_dir)
        raise

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

    # 古い世代をクリーンアップ
    grace_sec = env_int("INDEX_CLEANUP_GRACE_SEC", 300)
    cleanup_old_generations(gen_name, grace_sec)

    log_info("インデックス構築完了")


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
    for file in CACHE_DIR.glob("*.json*"):
        stem = file.name.split(".")[0]
        if stem not in valid_keys:
            try:
                file.unlink()
            except Exception:
                continue


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


def init_search_settings() -> Tuple[int, int, int]:
    budget = cpu_budget()
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
        log_info("スケジュール再構築完了")


def rebuild_fixed_cache():
    global fixed_cache_index
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
        params = SearchParams(entry["mode"], entry["range"], entry["space"])
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
            "folders": target_ids,
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
    prune_cache_dir(set(new_index.keys()))


def per_request_workers() -> int:
    max_workers = cpu_budget()
    env_workers = os.getenv("SEARCH_WORKERS", "").strip()
    if env_workers.isdigit():
        return max(1, int(env_workers))
    env_conc = os.getenv("SEARCH_CONCURRENCY", "").strip()
    if env_conc.isdigit():
        max_workers = max(1, max_workers // max(1, int(env_conc)))
    return max(1, max_workers)


@app.on_event("startup")
async def startup_event():
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
        log_info(f"検索実行モード: thread concurrency={concurrency} workers={workers} budget={budget}")
    scheme = "https" if (Path(os.getenv("CERT_DIR", "certs")) / "lan-cert.pem").exists() else "http"
    urls = [f"{scheme}://{ip}:{os.getenv('PORT', '80')}" for ip in (get_ipv4_addresses() or ["0.0.0.0"])]

    async def announce_access_urls():
        await asyncio.sleep(0.2)
        for url in urls:
            log_info(f"アクセスURL: {colorize_url(url)}")

    asyncio.create_task(announce_access_urls())
    rebuild_fixed_cache()
    asyncio.create_task(schedule_index_rebuild())


@app.get("/", response_class=FileResponse)
async def read_root():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UIが見つかりません")
    return FileResponse(index_file)


@app.get("/api/folders")
async def get_folders():
    return JSONResponse({"folders": describe_folder_state()})


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
    async with search_semaphore:
        async with rw_lock.read_lock():
            start_time = time.time()
            available_ids = set(folder_states.keys())
            target_ids = [f for f in req.folders if f in available_ids and folder_states[f].get("ready")]
            if not target_ids:
                raise HTTPException(status_code=400, detail="有効な検索対象フォルダがありません")

            params = SearchParams(req.mode, req.range_limit, req.space_mode)
            cache_key = cache_key_for(req.query, params, target_ids)

            # キャッシュからの取得を試行
            cached_result = _try_get_memory_cache(cache_key, target_ids, req.query, params)
            if cached_result:
                return cached_result
            cached_result = _try_get_fixed_cache(cache_key, target_ids, req.query, params)
            if cached_result:
                return cached_result

            keywords, norm_keyword_groups = build_query_groups(req.query, req.space_mode)
            raw_keywords = keywords
            keywords_for_response = keywords
            worker_count = search_worker_count
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
                )
        elapsed = time.time() - start_time
        log_info(
            f"検索完了: folders={len(target_ids)} results={len(results)} "
            f"mode={req.mode} workers={worker_count} 時間={elapsed:.2f}s"
        )
        payload = {
            "count": len(results),
            "results": results,
            "keywords": keywords_for_response,
            "folder_ids": normalize_folder_ids(target_ids),
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


@app.get("/api/health")
async def health():
    ready = all(m.get("ready") for m in folder_states.values()) if folder_states else False
    return {"status": "ok", "ready": ready}


# --- 実行ヘルパー ---
def run():
    """ローカル開発用エントリポイント."""
    import uvicorn
    import signal

    port = int(os.getenv("PORT", "80"))
    cert_dir = os.getenv("CERT_DIR", "certs")
    base_dir = Path(__file__).resolve().parent
    cert_path = (base_dir / cert_dir).resolve()
    cert_file = cert_path / "lan-cert.pem"
    key_file = cert_path / "lan-key.pem"

    ssl_kwargs = {}
    if cert_file.exists() and key_file.exists():
        ssl_kwargs = {
            "ssl_certfile": str(cert_file),
            "ssl_keyfile": str(key_file),
        }

    config = uvicorn.Config(
        "web_server:app",
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
