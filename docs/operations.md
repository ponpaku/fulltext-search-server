# 運用ドキュメント

System Version: 1.3.1

README は概要と使い方に絞り、運用・詳細設定は本ドキュメントにまとめます。

## 1. 運用方針（設定の考え方）
- 基本設定は `config.json` を推奨（項目の構造が見やすく、変更差分も追いやすい）
- `.env` は「起動/環境に依存する値」だけに限定するのが運用しやすい
- 優先順位: `.env / OS環境変数` > `config.json` > 既定値
- `.env` に空値 `VAR=` を指定した場合は未設定として扱われ、config.json が採用されます

## 2. config.json 設定一覧（型/意味/既定値）
`config.example.json` をコピーして `config.json` を作成します。
ここでは各項目の意味を一覧で示します（既定値は `config.example.json` の値）。

### search
| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| folders | 配列（{label, path}） | - | 検索対象フォルダ。`label` は表示名、`path` は実パス |
| folder_aliases | オブジェクト | - | SMB/NAS の表示名置換（表示のみ） |
| execution_mode | 文字列 | `thread` | 検索実行モード（`thread` / `process`） |
| process_shared | 真偽値 | `true` | processモードで共有mmapを使うか |
| cpu_budget | 数値 | `6` | 1リクエストのCPU予算 |
| concurrency | 数値 | `6` | 同時検索リクエスト上限 |
| workers | 数値 | `6` | 1リクエストあたりの最大ワーカー数 |

### front
| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| results_batch_size | 数値 | `100` | 追加取得時の件数 |
| range_default | 数値 | `0` | 文字数範囲の既定値 |
| space_mode_default | 文字列 | `jp` | 空白除去モード既定 |
| normalize_mode_default | 文字列 | `normalized` | 表記ゆれ既定 |
| results_scroll_threshold_px | 数値 | `200` | 追加取得の閾値（px） |
| history_max_items | 数値 | `30` | 履歴保存上限 |
| range_max | 数値 | `5000` | 文字数範囲の最大値 |
| heartbeat_interval_ms | 数値 | `35000` | 心拍の基本間隔（ms） |
| heartbeat_jitter_ms | 数値 | `10000` | 心拍ジッター（ms） |
| heartbeat_min_gap_ms | 数値 | `5000` | 直近送信からの最小間隔（ms） |
| heartbeat_interaction_gap_ms | 数値 | `15000` | 操作後の追加送信間隔（ms） |
| heartbeat_idle_threshold_ms | 数値 | `90000` | アイドル判定（ms） |
| heartbeat_fail_threshold | 数値 | `2` | 失敗判定回数 |
| heartbeat_stale_multiplier | 数値 | `2` | stale判定倍率 |
| health_check_interval_ms | 数値 | `5000` | health確認間隔（ms） |
| health_check_jitter_ms | 数値 | `3000` | healthジッター（ms） |

### heartbeat（サーバ）
`front.heartbeat_*` はクライアント送信間隔、`heartbeat.*` はサーバ側のTTL/上限です。

| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| ttl_sec | 数値 | `90` | アクティブ判定のTTL（秒） |
| max_clients | 数値 / null | `null` | 同時クライアント上限（`null` で自動算出、`0` は `1` に丸められる） |

### warmup
| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| enabled | 真偽値 | `true` | warmup を有効化 |
| idle_sec | 数値 | `1800` | アイドル判定（秒） |
| interval_sec | 数値 | `3600` | 再実行間隔（秒） |
| max_files | 数値 | `40` | 対象ファイル数上限 |
| head_mb | 数値 | `2` | 先頭ウォームアップ量（MB） |
| stride_mb | 数値 | `4` | ストライド量（MB） |
| max_mb | 数値 | `0` | 1ファイルの上限（MB、0=無制限） |

### query
| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| normalize | 文字列 | `nfkc_casefold` | 正規化方式 |

### query_stats
| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| ttl_days | 数値 | `30` | 統計の保持日数 |
| flush_sec | 数値 | `60` | フラッシュ間隔（秒） |

### cache
| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| fixed_min_count | 数値 | `10` | 固定キャッシュ対象の最低件数 |
| fixed_min_time_ms | 数値 | `500` | 固定キャッシュ対象の最低処理時間（ms） |
| fixed_min_hits | 数値 | `5000` | 固定キャッシュ対象の最低ヒット数 |
| fixed_min_kb | 数値 | `2000` | 固定キャッシュ対象の最低サイズ（KB） |
| fixed_ttl_days | 数値 | `7` | 固定キャッシュの保持日数 |
| fixed_max_entries | 数値 | `20` | 固定キャッシュ最大件数 |
| fixed_trigger_cooldown_sec | 数値 | `300` | 固定キャッシュ再生成の冷却時間（秒） |
| mem_max_mb | 数値 | `200` | メモリキャッシュ上限（MB） |
| mem_max_entries | 数値 | `200` | メモリキャッシュ最大件数 |
| mem_max_result_kb | 数値 | `2000` | 1結果の最大サイズ（KB） |
| compress_min_kb | 数値 | `2000` | 圧縮対象の最小サイズ（KB） |

### rebuild
| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| schedule | 文字列 | `03:00` | 再構築スケジュール |
| allow_shrink | 真偽値 | `true` | インデックス縮小を許可 |

### index
| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| keep_generations | 数値 | `3` | 保持世代数 |
| keep_days | 数値 | `30` | 保持日数 |
| max_bytes | 数値 | `0` | 全世代の最大サイズ（0=無制限） |
| cleanup_grace_sec | 数値 | `300` | 切替後猶予（秒） |
| store_normalized | 真偽値 | `true` | 正規化済みテキストを保存 |

### diff
| キー | 型 | 既定値 | 説明 |
| --- | --- | --- | --- |
| mode | 文字列 | `stat` | 差分判定モード（stat/fastfp/fullhash） |
| fast_fp_bytes | 数値 | `65536` | fastfp の読み取りバイト数 |
| full_hash_algo | 文字列 | `sha256` | fullhash のハッシュ方式 |
| full_hash_paths | 配列（文字列） | - | fullhash を強制するパス |
| full_hash_exts | 配列（文字列） | - | fullhash を強制する拡張子 |

## 3. .env（環境依存・秘密・一時上書き）
`.env` に書くのは次のような「環境依存」または「運用上の一時的上書き」に限定するのが推奨です。
heartbeat/warmup も `config.json` で設定可能なので、`.env` は上書き用途として利用します。

```env
# 起動ポート
# PORT=80

# SSL 証明書ディレクトリ（lan-cert.pem / lan-key.pem）
# CERT_DIR="certs"

# デバッグログ（1/0）
# SEARCH_DEBUG=1

# config.json のパス
# CONFIG_PATH="config.json"

```

## 4. 抽出の限界
- スキャンPDF（画像のみ）は本文を抽出できない場合があります（OCR非対応）
- パスワード付き/破損ファイルはスキップされる可能性があります
- 抽出失敗はログに出るため、該当ファイルを確認してください

## 5. 差分判定（DIFF_MODE / diff.mode）
差分検出は `config.json` の `diff.mode` で指定します（互換として `.env` の `DIFF_MODE` も有効）。

- `stat`（既定）: サイズ + mtime で判定（最速）
- `fastfp`: 先頭Nバイトの軽量ハッシュ（`diff.fast_fp_bytes`）
- `fullhash`: 全体ハッシュで厳密判定（`diff.full_hash_*`）

**注意**: SMB/NAS などで mtime が信用できない場合は `stat` だと見逃しが起きます。  
その場合は `fastfp`、または重要フォルダのみ `fullhash` を推奨します。

## 6. インデックス世代管理
- 起動時に全フォルダを走査して `indexes/gen_<uuid>/` に保存
- 変更ファイルのみ差分更新
- 再構築中も検索は継続稼働

### 世代管理
- 再構築時は `indexes/.build/gen_<uuid>/` で新世代を構築
- 完了後に `indexes/gen_<uuid>/` へ移動し、`current.txt` を更新
- 古い世代は保持ポリシーに従って自動削除
- 各世代に `manifest.json` を配置（メタデータ）

### 保持ポリシー
- `index.keep_generations`（env: `INDEX_KEEP_GENERATIONS`）: 保持する世代数（現在の世代と猶予期間中の世代を除く）
- `index.keep_days`（env: `INDEX_KEEP_DAYS`）: 作成日からの保持期間（日数、0=無制限）
- `index.max_bytes`（env: `INDEX_MAX_BYTES`）: **全世代の合計ディスク使用量の上限**（現在の世代と猶予期間中の世代を含む、0=無制限）
  - サイズ計算には猶予期間中の世代も含まれる
  - 超過した場合、猶予期間を過ぎた古い世代から順に削除される
  - 現在の世代と猶予期間中の世代は削除されないため、これらだけで上限を超える場合は削除されない
- `index.cleanup_grace_sec`（env: `INDEX_CLEANUP_GRACE_SEC`）: 世代切替後の猶予期間（秒、デフォルト: 300）
  - 猶予期間中の世代は削除されないが、サイズ計算には含まれる

### ロールバック手順
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

**注意**: ロールバック後は、現在の世代より新しい世代は自動削除されません。必要に応じて手動で削除してください。

## 7. キャッシュ整合性
- 固定キャッシュは `index_uuid` / `schema_version` を保存し、世代切替で自動無効化されます
- ロールバック時はキャッシュが無効化されるため、必要に応じて再構築します

## 8. 検索キャッシュの仕組み
- メモリキャッシュは LRU 方式で小さい結果を高速化
- 固定キャッシュは「頻出かつ重い検索」を対象にディスク保存（`cache/`）
- インデックス更新後に固定キャッシュを再生成して精度を維持
- 検索条件が閾値に達した場合は自動で固定キャッシュを再構築

## 9. 検索の並列化
- フォルダ単位は直列、ページ単位は並列
- 並列数は「アクティブクライアント数」で動的に配分
  - `SEARCH_WORKERS` は1リクエストあたりの上限
  - アクティブ数は心拍（TTL内の client_id）で推定
  - client_id はブラウザの localStorage に保存（同一ブラウザの複数タブは1クライアント扱い）
- 同時リクエスト数は `SEARCH_CONCURRENCY` で制御

## 10. 典型ケースの推奨設定例
### ローカルSSD（mtime 信頼可）
- `diff.mode`: `stat`
- `search.execution_mode`: `thread`
- `search.process_shared`: `true`

### SMB/NAS（mtime が不安定）
- `diff.mode`: `fastfp` または `fullhash`
- `diff.fast_fp_bytes`: 65536 など
- 重要フォルダのみ `diff.full_hash_paths` で `fullhash` 対象に

### 大規模共有（同時アクセス多め）
- `search.execution_mode`: `process`
- `search.process_shared`: `true`
- `search.cpu_budget` / `search.concurrency` / `search.workers` を控えめに調整

## 11. トラブルシューティング / FAQ
- 検索漏れがある: スキャンPDF（OCR非対応）/差分判定のモード/共有フォルダの更新時刻を確認
- 変更が反映されない: `diff.mode` を `fastfp` か `fullhash` に変更して再構築
- 遅い: `search.execution_mode` を `process` にする、`process_shared` を ON にする、NAS なら同時実行数を控えめにする
- 起動に失敗: `config.json` の存在と JSON 構文を確認
