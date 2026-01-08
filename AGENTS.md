# System Overview
System Version: 1.1.0  
AGENTS File Version: 1.0.0
- Stack: FastAPI backend with a static UI (`static/`) and in-memory search data.
- Inputs: folders from `SEARCH_FOLDERS`; optional SMB path parsing and display aliases via `SEARCH_FOLDER_ALIASES`.
- Indexing: extracts text from PDF/DOCX/TXT/CSV/XLSX/XLS and caches per-file text in gzip pickles under `indexes/`.
- Search: page-level scan with normalized text; supports AND/OR and optional range window; results include snippet/detail.

# Search Architecture
- Disk cache (`indexes/index_*.pkl.gz`): `{path -> {mtime, data{page->text}, high_acc}}`.
- Memory pages (`memory_pages`): flat list of entries per page with `raw`, `norm`, `norm_jp`, `norm_all`.
- Query flow: split query → normalize by `space_mode` → per-page substring match → snippet/detail extraction.

# PDF Extraction Behavior
- Uses pdfminer layout parsing; attempts to order text boxes by reading order.
- Multi-column PDFs can still produce mixed text; snippet now uses raw text for readability.
- `(cid:####)` noise and invisible separators are removed during normalization.

# Performance and Concurrency
- Indexing: parallelized with `ProcessPoolExecutor` for extraction; runs at startup only.
- Search: CPU-bound scanning; chunked per-folder when large; uses thread pool for page chunks.
- Concurrency model:
  - Read/Write lock: read for searches, write for startup indexing.
  - Semaphore: limits concurrent search requests (`SEARCH_CONCURRENCY`).
  - Worker budget per request: fixed at startup (`SEARCH_WORKERS` or `cpu_budget // SEARCH_CONCURRENCY`).
  - Folder-level search is serial; page-level is parallel (best for large folders).
  - Optional process mode: search execution can run in a ProcessPool (`SEARCH_EXECUTION_MODE=process`).

# UI Behavior
- Results are rendered in batches (100 items) with infinite scroll to avoid heavy DOM cost.
- View toggles for hit/file remain; highlight logic supports space-removed matches.
- Search options: AND/OR, range (AND only), space removal (default: Japanese-only).

# Configuration
- SEARCH_FOLDERS: `label=path` list; supports `;`, `,`, `|` separators and inline `#` comments.
- SEARCH_FOLDER_ALIASES: `host=alias` for display-only path replacement.
- SEARCH_CONCURRENCY: concurrent search request limit (default: CPU count).
- SEARCH_WORKERS: per-request worker count (optional; overrides auto).
- SEARCH_EXECUTION_MODE: `thread` or `process` for search execution.
- CERT_DIR: SSL cert directory (`lan-cert.pem`, `lan-key.pem`).

# Known Issues and Mitigations
- Mixed PDF text order: mitigated by layout-based ordering; still depends on source PDF.
- Large result rendering lag: fixed with batch rendering and infinite scroll.
- Connection reset logs: filtered to reduce noisy stack traces on Windows.

# Open Ideas (Issue Notes)
- Query cache exploration: track query frequency/latency and consider caching only high-frequency, heavy queries.
- Cache strategy: prefer in-memory LRU with size limits; store hit IDs instead of full results to reduce memory.
- Invalidation: clear cache on index rebuild or per-folder changes; avoid stale results.
- Optional warm queries: run a short list of known frequent queries at startup once patterns are known.

# Recent Changes Summary
- Removed morphological/inverted-index search due to precision regressions.
- Default space removal set to Japanese-only.
- Added read/write lock and search concurrency control with worker budgeting.
- Added SSL startup configuration with `CERT_DIR`.

# Change Log Policy
- `CHANGELOG.md` を日本語で更新してください。
- Keep System Version and File Version in each file.
