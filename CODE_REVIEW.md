# コードレビューレポート

**レビュー日**: 2026-01-07
**プロジェクト**: Full-Search-PDFs (全文検索サーバー)
**バージョン**: 1.0.0
**技術スタック**: FastAPI + Vanilla JS
**対象ファイル**: `web_server.py` (2,537行), `static/app.js` (824行), `static/index.html` (222行), `static/styles.css`
**想定利用環境**: 社内LAN / 家庭内LAN（信頼されたネットワーク）

---

## 前提条件

本レビューは、以下の利用環境を前提としています：

- **利用範囲**: 社内LAN / 家庭内LANに限定
- **ユーザー**: 信頼された組織内のメンバー
- **ネットワーク**: ルーター/ファイアウォールで外部からのアクセスを遮断
- **データソース**: 社内/家庭内の文書（PDF、Office）

---

## 1. 総合評価

| カテゴリ | 評価 | コメント |
|---------|------|----------|
| セキュリティ | ✅ | LAN内利用では適切な設計 |
| パフォーマンス | ✅ | キャッシュ・並列処理は良好 |
| コード品質 | ✅ | 現状の規模では適切 |
| 保守性 | ✅ | 単一ファイルだが関数分割は適切 |
| フロントエンド | ✅ | 良好な実装 |
| エラーハンドリング | ✅ | フォールトトレラントな設計 |

---

## 2. セキュリティ（LAN内利用を前提とした評価）

### ✅ 適切な設計

#### 2.1 CORS設定 (`web_server.py:1414-1420`)
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    ...
)
```
**評価**: LAN内では様々なデバイスから異なるIPアドレス経由でアクセスするため、妥当な設定です。

#### 2.2 Pickleの使用 (`web_server.py:1180-1183`)
**評価**: インデックスファイルはアプリ自身が生成するものであり、外部入力ではありません。安全です。

#### 2.3 入力検証 (`web_server.py:1391-1409`)
```python
class SearchRequest(BaseModel):
    query: str = Field(..., description="検索キーワード")
    mode: str = Field("AND", pattern="^(AND|OR)$")
    range_limit: int = Field(0, ge=0, le=5000)
    space_mode: str = Field("jp", pattern="^(none|jp|all)$")
```
**評価**: Pydanticによる適切なバリデーションが実装されています。

### 🟡 インターネット公開時に対応が必要

将来的にインターネットに公開する場合は、以下を検討してください：
1. CORS設定の制限（特定オリジンに限定）
2. 認証機能の追加
3. レート制限の実装

---

## 3. パフォーマンス

### ✅ 優れた設計

#### 3.1 正規表現のプリコンパイル (`web_server.py:60-63`)
```python
CID_PATTERN = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
INVISIBLE_SEPARATORS_PATTERN = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00a0\u202f]")
```
**評価**: モジュールレベルでコンパイル済みで、パフォーマンス最適化されています。

#### 3.2 LRUメモリキャッシュ (`web_server.py:1708-1765`)
```python
class MemoryCache:
    def __init__(self, max_entries: int, max_bytes: int, max_result_kb: int):
        self._data: OrderedDict[str, Dict] = OrderedDict()
```
**評価**: メモリ上限とエントリ数の両方を管理する適切なLRU実装です。

#### 3.3 並列処理 (`web_server.py:876-947`, `991-1138`)
**評価**: ThreadPoolExecutor/ProcessPoolExecutorを適切に使い分けており、大量データの検索を効率化しています。共有メモリ（mmap）によるプロセス間データ共有も実装されています。

#### 3.4 差分インデックス更新 (`web_server.py:1206-1298`)
**評価**: ファイルのmtimeをチェックし、変更があったファイルのみを再インデックスする効率的な実装です。

#### 3.5 無限スクロール (`app.js:618-643`)
```javascript
const appendNextBatch = () => {
  const slice = list.slice(start, end);  // 100件ずつ
  ...
};
```
**評価**: パフォーマンスを考慮したバッチレンダリングが適切に実装されています。

### 🟡 将来的な改善候補

#### 3.6 検索ロジックの冗長性 (`web_server.py:531-668` と `740-846`)
`search_text_logic` と `search_text_logic_shared` に類似ロジックが存在します。
**推奨**: 共通処理を抽出してDRY原則を適用（中長期課題）。

---

## 4. コード品質

### ✅ 良い設計

#### 4.1 型ヒントの活用
```python
def build_query_groups(query: str, space_mode: str) -> Tuple[List[str], List[List[str]]]:
def cache_key_for(query: str, params: "SearchParams", target_ids: List[str]) -> str:
```
**評価**: 主要な関数に型ヒントが付与されており、コードの可読性が高いです。

#### 4.2 例外処理のフォールトトレランス (`web_server.py:918-924`)
```python
for future in futures:
    try:
        res = future.result()
        if res:
            folder_results.extend(res)
    except Exception:
        continue
```
**評価**: 1つのファイルでエラーが発生しても他のファイルの検索を続行する妥当な設計です。`SEARCH_DEBUG=1` でログ出力可能です。

#### 4.3 設定の外部化
```python
env_int("CACHE_MEM_MAX_ENTRIES", 200)
env_int("CACHE_MEM_MAX_MB", 200)
os.getenv("SEARCH_EXECUTION_MODE", "thread")
```
**評価**: 設定値は`.env`から読み込む設計で、ハードコードを避けています。

#### 4.4 ログ出力 (`web_server.py:1660-1691`)
```python
def log_info(message: str):
def log_warn(message: str):
def log_success(message: str):
```
**評価**: 色付きログ出力で運用時の視認性が高いです。

### 🟡 将来的な改善候補

#### 4.5 マジックナンバー
```python
if len(entries_iter) >= 2000:  # web_server.py:898
detail_window = 2000  # web_server.py:648, 826
```
**推奨**: 定数として定義（優先度低）。

---

## 5. フロントエンド (app.js)

### ✅ 良い設計

#### 5.1 状態管理 (`app.js:7-23`)
```javascript
const state = {
  folders: [],
  selected: new Set(),
  mode: 'AND',
  keywords: [],
  ...
};
```
**評価**: 単一の状態オブジェクトで管理しており、理解しやすい設計です。

#### 5.2 クリップボード操作 (`app.js:55-104`)
```javascript
const copyTextToClipboard = async (text) => {
  const allowed = await requestClipboardPermission();
  if (!allowed) return false;
  ...
};
```
**評価**: パーミッション確認とフォールバック処理（execCommand）が適切に実装されています。

#### 5.3 テーマ切替 (`app.js:110-122`)
**評価**: localStorage でユーザー設定を永続化し、システム設定（prefers-color-scheme）も考慮しています。

#### 5.4 正規表現エスケープ (`app.js:436`)
```javascript
const escapeRegex = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
```
**評価**: ユーザー入力をハイライト正規表現に使用する際、適切にエスケープしています。

#### 5.5 モーダル管理 (`app.js:306-408`)
**評価**: aria属性を使用したアクセシビリティ対応が実装されています。

---

## 6. HTML/CSS

### ✅ 良い設計

#### 6.1 CSS変数の活用 (`styles.css`)
```css
:root {
  --bg: #ffffff;
  --surface: #f8fafc;
  --accent: #2563eb;
  ...
}
[data-theme="dark"] {
  --bg: #0f172a;
  ...
}
```
**評価**: ライト/ダークテーマをCSS変数で切り替えており、保守性が高いです。

#### 6.2 セマンティックHTML (`index.html`)
**評価**: `<header>`, `<section>`, `<label>`, `aria-*` 属性などを適切に使用しています。

#### 6.3 レスポンシブデザイン
**評価**: メディアクエリで適切にレイアウトを調整しています。

---

## 7. アーキテクチャ

### ✅ 適切な設計

#### 7.1 インデックス戦略
- 起動時に全フォルダを走査してインデックス構築
- 差分更新（mtime比較）で効率化
- gzip pickle形式で`indexes/`に永続化
- スケジュール再構築機能

#### 7.2 キャッシュ戦略
- **LRUメモリキャッシュ**: 最大200MB、小さい結果を高速化
- **固定ディスクキャッシュ**: 頻出かつ重い検索を`cache/`に保存
- **自動再構築**: クエリ条件が閾値到達時にトリガー

#### 7.3 検索の並列化
- フォルダ単位: 直列処理（ファイル破損対策）
- ページ単位: ThreadPoolExecutor（IO無し、CPU集約的）
- プロセスモード: ProcessPoolExecutor + 共有メモリ（mmap）

---

## 8. 解決済みの問題

#### ~~8.1 重複したIPアドレス取得関数~~ ✅
`run()` 関数内の未使用の重複定義を削除しました。

---

## 9. 将来的な改善候補（優先度低）

### 中長期（機能拡張時に検討）
1. ファイル分割によるモジュール化（現状2,537行で管理可能）
2. テストフレームワークの導入
3. マジックナンバーの定数化
4. 検索ロジックのDRY化

### インターネット公開時のみ必要
1. CORS設定の制限
2. 認証機能の追加
3. レート制限の実装

---

## 10. 結論

本プロジェクトは、社内LAN/家庭内LANでの利用を前提とした全文検索サーバーとして、**適切に設計・実装されています**。

### 主な強み
- **パフォーマンス**: 多層キャッシュ、並列処理、差分インデックス更新
- **堅牢性**: フォールトトレラントな例外処理
- **ユーザビリティ**: 無限スクロール、テーマ切替、キーボードショートカット
- **運用性**: 設定の外部化、色付きログ、スケジュール再構築

### 現状の評価
全体として、v1.0.0として十分な品質を持つコードベースです。LAN内利用の前提を踏まえると、セキュリティ上の懸念もありません。READMEにはLAN内利用の前提が明記されており、インターネット公開時に必要な対策も文書化されています。

---

*レビュー作成: Claude Code Review*
*最終更新: 2026-01-07*
