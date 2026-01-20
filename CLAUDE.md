# CLAUDE.md

System Version: 1.3.1

## 1. Project Context
- ローカル/社内フォルダの PDF/Office 文書を全文検索する FastAPI アプリ（UI は `static/` の静的配信）。
- 起動時にインデックスを作成し、検索はメモリ上のページ単位データを走査する設計。
- 検索は正規化テキストの部分一致（AND/OR、AND時の範囲指定）で高速化を優先。

## 2. Project Map
- `app.py`: エントリポイント（ASGI app）。
- `backend/`: API/検索/インデックス/キャッシュ/並列化の中核。
- `static/`: UI（`index.html`, `app.js`, `styles.css`）。
- `docs/operations.md`: 運用・設定の詳細ドキュメント。
- `indexes/`: 生成物。インデックス gzip pickle（コミットしない）。
- `cache/`: 生成物。固定キャッシュ等（コミットしない）。
- `backups/`: 参考スナップショット（コミットしない）。
- `run.bat` / `run.sh`: 起動スクリプト（pip install を毎回実行）。
- `.env` / `.env.example.txt`: 設定（`.env` はコミットしない）。
- `README.md` / `AGENTS.md`: 仕様・アーキテクチャの一次情報。

## 3. Standard Workflow
- 変更前に `README.md` と `AGENTS.md` を確認し、既存仕様と矛盾しないようにする。
- 変更後は `CHANGELOG.md` を日本語で更新。
- 生成物ディレクトリ（`indexes/`, `cache/`, `backups/`）は編集しない。
- 挙動や設定項目を変更した場合は、`README.md` / `docs/operations.md` / `AGENTS.md` / `CLAUDE.md` を適切に更新する。
- System Version を変更した場合は、`README.md` / `AGENTS.md` / `CLAUDE.md` / `CHANGELOG.md` を一致させる。

## 4. Common Commands
- 推奨起動（Linux/macOS）: `./run.sh`
- 推奨起動（Windows）: `run.bat`
- 直接起動: `uvicorn app:app --host 0.0.0.0 --port 80`
- 依存導入: `python -m pip install -r requirements.txt`

## 5. Coding Standards
- 既存の関数分割と例外処理パターンを尊重し、局所的な変更に留める。
- 文字コードは基本 ASCII。日本語が必要な場合のみ既存の文体に合わせて記述。
- 設定値は `.env` から読み込む前提で直書きしない（`os.getenv` を使用）。

## 6. Testing
- テストフレームワークは未整備。必要なら追加方針を先に合意する。

## 7. Tooling / Scripts / MCP
- `run.bat` / `run.sh` が標準の起動経路（venv作成 + pip install を含む）。
- 検索実行モード: `SEARCH_EXECUTION_MODE=thread|process`。
  - `process` では `SEARCH_PROCESS_SHARED=1`（既定）で共有 mmap を利用。

## 8. Safety / Secrets
- `.env`、`certs/`（`lan-cert.pem` / `lan-key.pem`）、`indexes/`、`cache/`、`backups/` はコミットしない。
- 検索対象ファイル（PDF/DOCX/TXT/CSV/XLSX/XLS）は機密の可能性があるため、コード変更時に同梱しない。
- `.env` と `config.json` の中身は閲覧しない。
- `indexes/` 配下のインデックスファイルの中身は閲覧しない。
