# コードレビューレポート

**レビュー日**: 2026-01-07
**プロジェクト**: Full-Search-PDFs (全文検索サーバー)
**バージョン**: 1.0.0
**技術スタック**: FastAPI + Vanilla JS
**対象ファイル**: `web_server.py`, `static/app.js`, `static/index.html`, `static/styles.css`

---

## 1. セキュリティ

### 🔴 重大な問題

#### 1.1 CORS設定が過度に緩い (`web_server.py:1414-1420`)
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
**問題**: `allow_origins=["*"]` と `allow_credentials=True` の組み合わせは、CSRF攻撃のリスクを高めます。
**推奨**: 本番環境では特定のオリジンに制限するか、環境変数で設定可能にする。

#### 1.2 Pickleの使用 (`web_server.py:1180-1183`)
```python
with gzip.open(idx_path, "rb") as f:
    data = pickle.load(f)
```
**問題**: Pickleは任意コード実行の脆弱性があります。悪意のあるインデックスファイルが挿入された場合、コードが実行される可能性があります。
**推奨**: JSONまたはMessagePackなど、より安全なシリアライゼーション形式への移行を検討。

#### 1.3 パストラバーサルの潜在的リスク (`web_server.py:2304-2337`)
```python
rel = os.path.relpath(path, base_path)
files.append({"path": path, "relative": rel, ...})
```
**問題**: APIレスポンスに絶対パスを含めており、サーバー構造が露出する可能性があります。
**推奨**: 内部パスは返さず、相対パスまたはIDのみを返す。

### 🟡 中程度の問題

#### 1.4 入力検証の不足 (`web_server.py:1399-1403`)
```python
@field_validator("query")
def validate_query(cls, v: str) -> str:
    if not v.strip():
        raise ValueError("検索キーワードを指定してください")
    return v
```
**問題**: 検索クエリに対するサニタイズが不十分。正規表現インジェクションのリスクがあります。
**推奨**: 検索クエリの長さ制限と特殊文字のエスケープ処理を追加。

---

## 2. パフォーマンス

### 🟢 良い設計

#### 2.1 メモリキャッシュ (`web_server.py:1708-1765`)
LRUキャッシュの実装は適切で、メモリ上限とエントリ数の両方を管理しています。

#### 2.2 並列処理 (`web_server.py:876-947`)
ThreadPoolExecutor/ProcessPoolExecutorを適切に使い分けており、大量データの検索を効率化しています。

### 🟡 改善の余地

#### 2.3 重複したIPアドレス取得関数 (`web_server.py:134-170` と `2533-2569`)
**問題**: `get_ipv4_addresses()` 関数がファイル内で2回定義されています。
**推奨**: 共通関数として1箇所にまとめる。

#### 2.4 大量の正規表現コンパイル (`web_server.py:60-64`)
```python
CID_PATTERN = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
INVISIBLE_SEPARATORS_PATTERN = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00a0\u202f]")
```
**良い点**: 正規表現はモジュールレベルでコンパイル済みで、パフォーマンス最適化されています。

#### 2.5 検索ロジックの冗長性 (`web_server.py:531-668` と `740-846`)
**問題**: `search_text_logic` と `search_text_logic_shared` にほぼ同じロジックが存在します。
**推奨**: 共通処理を抽出してDRY原則を適用。

---

## 3. コード品質

### 🔴 重大な問題

#### 3.1 グローバル状態の過剰な使用 (`web_server.py:1502-1532`)
```python
host_aliases = parse_host_aliases()
configured_folders = parse_configured_folders()
folder_states: Dict[str, Dict] = {}
memory_indexes: Dict[str, Dict[str, Dict]] = {}
memory_pages: Dict[str, List[Dict]] = {}
# ... 30以上のグローバル変数
```
**問題**: 30以上のグローバル変数が存在し、状態管理が複雑化しています。テストが困難になり、スレッドセーフティの問題を引き起こす可能性があります。
**推奨**: 依存性注入パターンまたはクラスベースの状態管理に移行。

#### 3.2 例外の黙殺 (`web_server.py:919-924`, `1045-1048`, など)
```python
except Exception:
    continue
```
**問題**: 例外を無視しており、デバッグが困難になります。
**推奨**: 少なくともログを出力し、問題を追跡可能にする。

### 🟡 中程度の問題

#### 3.3 関数の責務が多すぎる (`web_server.py:2222-2288`)
`startup_event` 関数が約70行で、複数の責務（初期化、インデックス構築、スケジューラ起動など）を担っています。
**推奨**: 各責務を個別の関数に分割。

#### 3.4 マジックナンバーの多用
```python
if len(entries_iter) >= 2000:  # web_server.py:898
detail_window = 2000  # web_server.py:826
snippet_start = max(0, raw_hit_pos - 40)  # web_server.py:642
snippet_end = min(len(raw_for_positions), snippet_start + 160)  # web_server.py:643
```
**推奨**: 定数として定義し、意味のある名前を付ける。

#### 3.5 型ヒントの不整合
```python
def build_index_for_folder(...) -> Tuple[Dict[str, Dict], Dict, Dict[str, str]]:
```
**問題**: 返り値の2番目の `Dict` に詳細な型ヒントがありません。
**推奨**: `Tuple[Dict[str, Dict], Dict[str, Any], Dict[str, str]]` のように具体化。

---

## 4. フロントエンド (app.js)

### 🟢 良い設計

#### 4.1 状態管理 (`app.js:7-23`)
```javascript
const state = {
  folders: [],
  selected: new Set(),
  mode: 'AND',
  // ...
};
```
単一の状態オブジェクトで管理しており、理解しやすい設計です。

#### 4.2 無限スクロール (`app.js:618-643`)
パフォーマンスを考慮したバッチレンダリングが適切に実装されています。

### 🟡 改善の余地

#### 4.3 XSS対策の不足 (`app.js:446-468`)
```javascript
const highlightText = (text, keywords) => {
  // ...
  result = result.replace(regex, match => `<span class="highlight">${match}</span>`);
```
**問題**: ハイライト処理でHTMLをエスケープせずに直接挿入しています。
**推奨**: `textContent` を使用するか、適切なエスケープ処理を追加。

#### 4.4 エラーハンドリングの改善 (`app.js:716-742`)
```javascript
} catch (err) {
  resultsEl.innerHTML = `...${err.message}...`;
}
```
**問題**: エラーメッセージを直接HTMLに挿入しており、XSSの可能性があります。
**推奨**: `textContent` を使用。

---

## 5. CSS (styles.css)

### 🟢 良い設計

#### 5.1 CSS変数の活用 (`styles.css:7-50`)
ライト/ダークテーマを CSS変数で切り替えており、保守性が高いです。

#### 5.2 レスポンシブデザイン (`styles.css:1041-1137`)
メディアクエリで適切にレイアウトを調整しています。

### 🟡 改善の余地

#### 5.3 ハードコードされた色 (`styles.css:409`)
```css
.toggle-btn.active {
  background: var(--accent);
  color: #fff;  /* ハードコード */
}
```
**推奨**: `#fff` も CSS変数化して一貫性を保つ。

---

## 6. 構造的な問題

### 🔴 重大な問題

#### 6.1 ファイルサイズ
`web_server.py` が2,575行で1ファイルに全ロジックが集約されており、保守性が低下しています。

**推奨構成**:
```
src/
├── main.py              # FastAPIアプリのエントリポイント
├── routes/
│   ├── search.py        # 検索API
│   └── folders.py       # フォルダAPI
├── services/
│   ├── indexer.py       # インデックス構築
│   ├── search_engine.py # 検索ロジック
│   └── cache.py         # キャッシュ管理
├── extractors/
│   ├── pdf.py           # PDF抽出
│   ├── docx.py          # Word抽出
│   └── excel.py         # Excel抽出
└── utils/
    ├── text.py          # テキスト正規化
    └── network.py       # ネットワークユーティリティ
```

---

## 7. テスタビリティ

### 🔴 重大な問題

#### 7.1 テストの欠如
テストコードが存在しません。グローバル状態への依存が多く、ユニットテストが困難です。

**推奨**:
1. 依存性注入パターンの導入
2. ビジネスロジックの純粋関数化
3. pytest + pytest-asyncio でテスト環境を整備

---

## 8. 総合評価

| カテゴリ | 評価 | コメント |
|---------|------|----------|
| セキュリティ | ⚠️ | CORS、Pickle、入力検証に改善が必要 |
| パフォーマンス | ✅ | キャッシュ・並列処理は良好 |
| コード品質 | ⚠️ | グローバル状態の過剰使用が課題 |
| 保守性 | ⚠️ | ファイル分割と構造化が必要 |
| フロントエンド | ✅ | 概ね良好、XSS対策を強化 |
| テスタビリティ | ❌ | テスト未整備 |

---

## 9. 優先度別改善推奨事項

### 即座に対応すべき (Critical)
1. CORS設定の見直し
2. XSSの潜在的脆弱性の修正
3. 例外の適切なロギング

### 短期的に対応 (High)
1. 検索クエリの入力検証強化
2. 重複関数の統合
3. APIレスポンスからの内部パス除去

### 中長期的に対応 (Medium)
1. ファイル分割によるモジュール化
2. テストフレームワークの導入
3. Pickle から JSON/MessagePack への移行
4. グローバル状態のクラスベース管理への移行

---

*レビュー作成: Claude Code Review*
