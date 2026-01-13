# Full-Search-PDFs

System Version: 1.1.11
README File Version: 1.1.11

ローカル/社内フォルダ内の PDF/Office 文書を全文検索する FastAPI アプリです。
UI は `static/` 配下で提供されます。

> **注意**: 本アプリは社内LAN/家庭内LANでの利用を前提として設計されています。
> インターネットに公開する場合は、認証機能の追加やCORS設定の見直しなど、追加のセキュリティ対策が必要です。

## 主要機能
- PDF/DOCX/TXT/CSV/XLSX/XLS のテキスト抽出・検索
- AND/OR + 文字数範囲指定（AND時）
- 和文のみ空白除去（デフォルト）
- 表記ゆれ吸収（既定、厳格は最小整形）
- 結果の無限スクロール（100件ずつレンダリング）
- SMB/NAS の表示名置換（表示のみ）
- SSL 起動（証明書を配置すれば自動）
- 起動時インデックス構築 + スケジュール再構築
- 検索キャッシュ（メモリ/LRU + 固定キャッシュ）

## 起動
```bash
python web_server.py
```

`.env` を使う場合は同ディレクトリに配置してください。

## 簡易起動スクリプト
### Windows
- `run.bat` を実行

### Linux / macOS
```bash
chmod +x run.sh
./run.sh
```

## 使い方（UI）
- 検索: キーワード入力 → 対象フォルダ選択 → 検索ボタン
- 絞り込み: 検索結果の「絞り込み」からフォルダ/ファイル形式でローカル絞り込み
  - フィルタ未選択は全件表示
- 表記ゆれ: 既定は「ゆらぎ吸収」。厳格は改行/不可視のみ整える
  - 環境設定により厳格へフォールバックする場合あり
- 検索履歴: 右上の履歴アイコンから表示（最大30件、ピン留め可）
- CSV出力: 検索結果がある時のみ「CSV」ボタンが表示

## NAS/共有フォルダ利用時の注意
- 事前にOS側で共有フォルダを認証・マウントしてから起動してください。
- アプリ側では資格情報を扱わず、OSの認証済みパスをそのまま利用します。

## タスクスケジューラー登録例（Windows）
- 基本: アクションは `run.bat` を指定
- コンソールを表示したい場合:
  - プログラム/スクリプト: `C:\Windows\System32\cmd.exe`
  - 引数の追加: `/k ""C:\path\to\run.bat""`
  - 開始（オプション）: `C:\path\to\`（run.bat のフォルダ）

## 設定（.env）
```env
# 検索対象フォルダ（ラベル=パス、1行で指定）
# Windowsは C:\Users\... をそのまま記載可（\ のエスケープ不要）
SEARCH_FOLDERS="規程=C:\data\docs,議事録=//192.168.30.10/share/minutes"
# 複数行に分割する設定は非対応（最初の1行のみ読み取り）
# OS環境変数が設定済みの場合はそちらが優先されます

# NAS名表示用（表示のみ）
SEARCH_FOLDER_ALIASES="192.168.30.10=landisk-fukyo"

# 検索同時実行数（デフォルト: CPU数）
SEARCH_CONCURRENCY=6

# 1検索あたりの並列ワーカー数（省略時は自動計算）
SEARCH_WORKERS=6

# CPUコア数の上書き（開発機と本番機が違う場合）
SEARCH_CPU_BUDGET=6

# ハートビートのTTL（アクティブ判定、秒）
HEARTBEAT_TTL_SEC=90

# アクティブクライアント数の上限（未指定ならワーカー予算に準拠）
HEARTBEAT_MAX_CLIENTS=6

# 検索実行モード（thread/process）
SEARCH_EXECUTION_MODE=thread

# SSL 証明書ディレクトリ（lan-cert.pem / lan-key.pem）
CERT_DIR="certs"

# クエリ統計の保持期間（日）
QUERY_STATS_TTL_DAYS=30

# クエリ統計の保存間隔（秒）
QUERY_STATS_FLUSH_SEC=60

# 固定キャッシュの条件（頻度・重さ）
CACHE_FIXED_MIN_COUNT=10
CACHE_FIXED_MIN_TIME_MS=500
CACHE_FIXED_MIN_HITS=5000
CACHE_FIXED_MIN_KB=2000

# 固定キャッシュの維持期間（日）
CACHE_FIXED_TTL_DAYS=7

# 固定キャッシュの最大件数
CACHE_FIXED_MAX_ENTRIES=20

# 固定キャッシュの自動再構築の間隔（秒）
CACHE_FIXED_TRIGGER_COOLDOWN_SEC=300

# メモリキャッシュの上限
CACHE_MEM_MAX_MB=200
CACHE_MEM_MAX_ENTRIES=200
CACHE_MEM_MAX_RESULT_KB=2000

# 固定キャッシュの圧縮閾値（KB以上はgzip保存）
CACHE_COMPRESS_MIN_KB=2000

# クエリ正規化モード（off/nfkc_casefold） ※推奨: nfkc_casefold
QUERY_NORMALIZE=nfkc_casefold

# インデックスに正規化済みテキストを保持（0/1） ※推奨: 1
INDEX_STORE_NORMALIZED=1

# インデックス再構築スケジュール（例: 03:00 or 12h）
REBUILD_SCHEDULE="03:00"
```

## インデックス
- 起動時に全フォルダを走査して `indexes/gen_<uuid>/` に保存
- 変更ファイルのみ差分更新
- スケジュール指定があれば自動で再構築
- 世代管理により、再構築中も検索が継続稼働
- 保持ポリシー（世代数/日数/容量）により古い世代を自動削除

### 世代管理
- 再構築時は `indexes/.build/gen_<uuid>/` で新世代を構築
- 完了後に `indexes/gen_<uuid>/` へ移動し、`current.txt` を更新
- 古い世代は保持ポリシーに従って自動削除
- 各世代に `manifest.json` を配置（メタデータ）

### 保持ポリシーの詳細
- `INDEX_KEEP_GENERATIONS`: 保持する世代数（現在の世代と猶予期間中の世代を除く）
- `INDEX_KEEP_DAYS`: 作成日からの保持期間（日数、0=無制限）
- `INDEX_MAX_BYTES`: **全世代の合計ディスク使用量の上限**（現在の世代と猶予期間中の世代を含む、0=無制限）
  - サイズ計算には猶予期間中の世代も含まれる
  - 超過した場合、猶予期間を過ぎた古い世代から順に削除される
  - 現在の世代と猶予期間中の世代は削除されないため、これらだけで上限を超える場合は削除されない
- `INDEX_CLEANUP_GRACE_SEC`: 世代切替後の猶予期間（秒、デフォルト: 300）
  - 猶予期間中の世代は削除されないが、サイズ計算には含まれる

### ロールバック手順
問題が発生した場合、手動で前の世代に戻すことができます：

1. サーバーを停止
2. `indexes/` ディレクトリ内の世代を確認：
   ```bash
   ls -lt indexes/gen_*
   ```
3. 戻したい世代の UUID を確認（例: `1704715496_abc12345`）
4. `current.txt` を手動で編集：
   ```bash
   echo "1704715496_abc12345" > indexes/current.txt
   ```
5. サーバーを再起動

**注意**: ロールバック後は、現在の世代より新しい世代は自動削除されません。
必要に応じて手動で削除してください。

## 検索キャッシュ
- メモリキャッシュは LRU 方式で小さい結果を高速化
- 固定キャッシュは「頻出かつ重い検索」を対象にディスク保存（`cache/`）
- インデックス更新後に固定キャッシュを再生成して精度を維持
- 検索条件が閾値に達した場合は自動で固定キャッシュを再構築

## 検索の並列化
- フォルダ単位は直列、ページ単位は並列
- 並列数は「アクティブクライアント数」で動的に配分
  - `SEARCH_WORKERS` は1リクエストあたりの上限
  - アクティブ数は心拍（TTL内の client_id）で推定
  - client_id はブラウザの localStorage に保存（同一ブラウザの複数タブは1クライアント扱い）
- 同時リクエスト数は `SEARCH_CONCURRENCY` で制御

## 注意点
- Windows の process モードは spawn のため、ワーカー数に応じてメモリが増えます
- `SEARCH_PROCESS_SHARED=1`（既定）ではページ本文が共有 mmap になり増加は抑制されます
- PDFのレイアウトによっては抽出順序が混ざる場合があります

## 変更履歴とバージョン運用
- 変更ごとに `CHANGELOG.md` を日本語で更新してください。
- 各ファイルには System Version と File Version を記載しています。
