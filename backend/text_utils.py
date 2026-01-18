"""Text normalization and splitting utilities."""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

from .config import DETAIL_CONTEXT_PREFIX, DETAIL_WINDOW_SIZE

SYSTEM_VERSION = "1.3.0"

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
    """Normalize text for keyword search."""
    if not text:
        return ""
    text = strip_cid(text)
    text = unicodedata.normalize("NFKC", text)
    text = normalize_invisible_separators(text)
    text = text.lower()
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_text_minimal(text: str) -> str:
    """Minimal normalization: newlines and invisible characters only."""
    if not text:
        return ""
    text = strip_cid(text)
    text = normalize_invisible_separators(text)
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_nfkc_casefold(text: str, compress_spaces: bool = True) -> str:
    """NFKC + casefold normalization."""
    if not text:
        return ""
    text = strip_cid(text)
    text = unicodedata.normalize("NFKC", text)
    text = normalize_invisible_separators(text)
    text = text.casefold()
    text = re.sub(r"[\n\t\r]", " ", text)
    if compress_spaces:
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


def build_detail_text(raw_text: str, raw_hit_pos: int) -> str:
    detail_pos = raw_hit_pos if raw_hit_pos != -1 else 0
    detail_start = max(0, detail_pos - DETAIL_CONTEXT_PREFIX)
    detail_end = min(len(raw_text), detail_start + DETAIL_WINDOW_SIZE)
    detail_text_raw = raw_text[detail_start:detail_end]
    return normalize_detail_text(detail_text_raw)


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
