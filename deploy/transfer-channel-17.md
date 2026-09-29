# 高速転送チャネル「.17」— 現状アーカイブ & ffformer-PVC 着地計画

> **✅ 2026-09-29 実施済み(§3 の推奨案を適用)。** HUCC+RC 再起動で extuser が過去の
> アップロード/結果を見られなくなった障害の復旧として、`.17` を本番データ経路へ移行した:
> 1. **pod は自分の PVC を索引**(`deploy/server.py` の `_scan_local_*`、SSH 廃止)→ 結果 10 件が復活。
> 2. **ffformer-infer-base に rclone webdav サイドカー追加**(`ffformer-pvc` の `results` を
>    subPath マウント、`:8081`、認証ユーザ `ffxfer`。パスワードは Deployment spec 内=live が正)。
> 3. **`svc/bw17`(VIP `172.31.229.17`)を ffformer pod:8081 に向け替え** → `.17` は now ffformer PVC を指す。
> 4. **grand1 から meta.json 16 件を `.17` 経由で PVC `_imports/` に backfill** → データセット 16 件が復活。
>
> 以降、grand1 は `http://133.50.39.17:443`(ユーザ `ffxfer`)へ push すると ffformer PVC の
> `results/` に着地する。**残タスクは末尾「§5 残り」**(新規ジョブの結果回送の反転)。
> 旧 bw17 Deployment(emptyDir + bwpass)は参照されなくなった(§0 の manifest はアーカイブ)。


grand1(HPC)⇄ クラウド の高速データ経路。1GbE 管理網(172.31.20.1 eno1、上限
119 MB/s)の代替として「サービス公開申請」で開通させた専用の公開IPペア。

- **公開IP `133.50.39.17`** ← grand1 から接続する外向きの入口
- **中間VIP `172.31.229.17`** ← NAT の内側、MetalLB が bw17 Service に割当
- 承認済みファイアウォール: `133.50.40.36-40 → :443/TCP`(= grand1 等ログインノードの出網元IP)

> **方向の鉄則**: 転送は必ず **grand1 が client として能動発起**する。
> pod→grand1 は eno1(1GbE)しか到達できず、bond0 / IB / grand1公開IP は FW で塞がれている。
> よって「pod が grand1 へ push」ではなく「grand1 が クラウドの入口(.17)へ push/pull」。

---

## §1 現状(2026-09-29 grand1 から実測・検証済み)

| 項目 | 結果 |
|---|---|
| TCP `133.50.39.17:443` | **OPEN** |
| プロトコル | **平文 HTTP**(TLS ではない。`https://`+`-k` は `packet length too long` で失敗) |
| サービス | rclone WebDAV、`Server: rclone/v1.75.1`(bw17 pod、image `rclone/rclone:latest`) |
| 資格情報 | `bw:bwpass`(★テスト用のまま。本番化時に要差し替え) |
| 中身 | `/big.bin`(5 GiB、測定用ダミー。initContainer が起動毎に再生成) |
| 単流下行スループット | **≈335–342 MB/s ≒ 2.74 Gbps**(curl / rclone cat 実測)|
| 8スレッド下行 | **≈6 Gbps / 734–850 MB/s**(2026-08-13 実測、`--multi-thread-streams 8`)|
| 中間VIP `172.31.229.17` | ✗ TCP 不達(grand1→bond0→VIP 経路は依然不通。**公開IP を使うこと**)|

いずれも 1GbE 管理網(119 MB/s)を大きく上回る。法定停電の影響も受けない。

### 検証済みコマンド(grand1 で今すぐ動く)

grand1 の rclone は **v1.57(古い)**。注意点2つ:
1. インライン接続文字列(`:webdav,url=...:`)は URL の `//` を誤解析するので **`--webdav-*` フラグ形式**を使う。
2. パスワードは **`rclone obscure` で難読化**して渡す(`rclone obscure` は毎回異なる値を出すが、いずれも有効)。

```bash
URL=http://133.50.39.17:443
PASS=$(rclone obscure bwpass)

# 一覧
rclone lsl :webdav: --webdav-url $URL --webdav-user bw --webdav-pass "$PASS"

# 下行スループット測定(単流、前 2GB を捨て読み)
curl -sS -u bw:bwpass -r 0-1999999999 $URL/big.bin -o /dev/null \
  -w "%{speed_download}B/s\n"

# 下行(8スレッド、ファイルをディスクへ)
rclone copy :webdav:/big.bin /dev/shm/ \
  --webdav-url $URL --webdav-user bw --webdav-pass "$PASS" \
  --multi-thread-streams 8 -P

# 上行(PUT)。v1.57 の WebDAV PUT は多スレッド非対応 → 単流 ≈1.5 Gbps 止まり。
#   ユーザ空間に新しい rclone を入れれば上行も多スレッド化できる。
rclone copy /dev/shm/big.bin :webdav: \
  --webdav-url $URL --webdav-user bw --webdav-pass "$PASS" -P
```

サーバ側(クラウド)の現物マニフェスト: **`deploy/bw17-webdav.yaml`**。

---

## §2 いま解けていない核心の制約

**bw17 の `/data` は emptyDir(揮発)で、ffformer-pvc とは別のディスク。**
→ grand1 から `.17` へデータを送っても **ffformer pod のストレージには入らない**。
現時点の `.17` は「転送速度が出ることの実証機」であって、まだ本番のデータ経路ではない。

さらに決定的な制約:

> **`ffformer-pvc` は `ReadWriteOnce`(RWO)**、storageClass `exascaler`、100Gi。
> RWO は原則 **同時に1つの pod(1ノード)** しかマウントできない。
> ffformer-infer-base が既にマウント中なので、**別 pod の bw17 から同じ PVC は同時マウントできない。**

このため「bw17 の emptyDir を ffformer-pvc に差し替えるだけ」では動かない。

---

## §3 計画 — 転送を ffformer-PVC に着地させる

### 推奨: WebDAV を **ffformer-infer-base の pod 内サイドカー**として動かす

同一 pod なら RWO PVC を自然に共有できる(スケジューリングの小細工不要)。grand1 が
`.17` へ push したファイルが、そのまま ffformer の `/workspace/data`(results / _imports)に着地する。

手順:
1. **ffformer-infer-base Deployment に rclone コンテナを追加**。既存の `data-vol`(=ffformer-pvc)
   を `/workspace/data` にマウントし、`serve webdav /workspace/data --addr :8080` で待受。
   - 書込み先を絞るなら `--webdav`… ではなくマウントを subPath(例 `_imports`)にする。
   - `bw:bwpass` を **Secret 化した強い資格情報**に差し替える(§4)。
2. **`.17` を ffformer pod に向け替え**: `svc/bw17` の `selector` を ffformer pod のラベルに、
   `targetPort` をサイドカーのポートに変更(承認済み FW とVIP `172.31.229.17` をそのまま再利用)。
   - もしくは bw17 Service はそのまま残し、ffformer 用に別 LB を建てる案もあるが、
     新規VIP+FW申請が要るので **既存 .17 の付け替え**が最短。
3. **`hpc_backend` / PBS の転送方向を反転**: 現行の「pod→HPC SSH(1GbE)」を
   「**grand1 が推論後の結果を `.17`(=ffformer PVC)へ rclone push**」に置換。
   - 入力(アップロード)側も同様に、pod ではなく grand1 側が `.17` から取得する形へ寄せる。
   - これで法定停電で管理網が絞られても本番経路が生き残る。

### 代替案(非推奨)
- **bw17 を ffformer と同ノードに co-locate + emptyDir→ffformer-pvc**: RWO を同一ノードの
  2 pod で共有できるかは exascaler CSI 次第で不確実。サイドカー案の方が確実。
- **bw17 に専用 RWX PVC を持たせ、pod 内で ffformer-pvc へ rsync**: ホップが増える。

### やること TODO
- [x] ffformer-infer-base に rclone サイドカー追加(ffformer-pvc の results を subPath マウント、:8081)
- [x] `svc/bw17`(=VIP .17)を ffformer pod へ向け替え(targetPort 8081)
- [x] pod 側の索引を PVC ローカル走査へ(server.py `_scan_local_*`)
- [x] grand1 から meta.json を `.17` 経由で backfill(データセット復活)
- [ ] **認証を Secret 化**(現在 `ffxfer:<pass>` を Deployment spec 内にインライン。
  `Secret-Store Writes` ガードで自動作成不可 → 要ユーザの kubectl 権限。§4 参照)
- [ ] `hpc_backend` の結果回送を grand1-initiated rclone push に反転(→ §5)
- [ ] grand1 にユーザ空間の新しい rclone を導入(上行の多スレッド化。現状 v1.57)
- [ ] 旧 bw17 Deployment を scale=0 or 削除(現在は参照されず遊休、big.bin 5GiB 保持)

---

## §5 残り — 新規ジョブの本番フロー反転(未実施)

今回の障害復旧で「**索引=PVCローカル**」「**既存データの着地=.17 経由**」は移行済み。ただし
**新規推論ジョブの往復はまだ旧経路(pod→HPC SSH, 1GbE)**のまま:

- 現状: `/import` が pod→HPC へ SFTP アップロード → `hpc_backend` が pod から qsub →
  完了後 pod が結果を SSH で pull(`fetch_result_bundle`)。管理網 SSH に依存。
- あるべき姿: grand1(=HPC ログインノード)側が主導。ジョブ完了後、**PBS ジョブ末尾で
  grand1 が結果一式を `.17`(ffxfer webdav)へ rclone push** → PVC `results/<id>/` に着地 →
  pod は `_scan_local_results()` で拾う。入力アップロードも同様に grand1 主導へ寄せる。
- これで法定停電で管理網が絞られても本番経路が生き残る。`hpc_backend.py` の
  `_submit_poll_download` / `fetch_result_bundle` と PBS テンプレートの改修が必要。

---

## §4 セキュリティ / 運用メモ
- 現在の `bw:bwpass` は**平文 HTTP 上の固定テスト資格情報**。本番化前に必ず差し替え、
  可能なら公開側で TLS 終端(ffformer の nginx-tls と同じ流儀)を検討。
- FW は `133.50.40.36-40 → :443/TCP` に限定済み(超算ログインノードのみ)。
- kubectl の Rancher トークンは失効する(`system:unauthenticated` 403)→ farm.hucc… で再取得。
- 関連: 生産アーキテクチャ全体は本 `deploy/` 配下と、K8s→HPC の 1GbE 経路の事実を参照。
