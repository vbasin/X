# Data Schemas

> **Source of truth for**: Parquet column specification, TimescaleDB table schemas, manifest format, training pipeline expectations

---

## 1. Parquet Feature Files

### 1.1 File Naming and Location

Canonical 5s research cache (ADR-079):

```
$VTECH_DATA_DIR/sources/features/timeframe=5s/symbol=NQ/year=YYYY/month=MM/features_5s_partN.parquet
```

Leftover day files (not written after ADR-081; operator may delete):

```
$VTECH_DATA_DIR/sources/features/parquet_5s/features_YYYY-MM-DD.parquet
```

### 1.2 Dimensions

| Property | Value |
|----------|-------|
| Rows | **1,080** (360 bars/hr × 3hr session: 08:00–11:00 CST) |
| Columns | **185** total |
| Index / datetime | Hive: column `bar_end_ts` (UTC). Loaders set DatetimeIndex name `ts`. Day files: unnamed UTC index. |
| Feature columns | **183** (`float32`) |
| Metadata columns | **2** (`string`) — excluded from training by trainer's `_`-prefix filter |

### 1.3 Column Groups

| Column(s) | Dtype | Notes |
|-----------|-------|-------|
| `bp_*` (24) | float32 | Book Pressure |
| `of_*` (17) | float32 | Order Flow (includes `of_ofi` / `of_ofi_zscore_60`, ADR-068) |
| `tl_*` (16) | float32 | Trade Location |
| `ss_*` (3) | float32 | Same-Side Runs |
| `lt_*` (4) | float32 | Large-Trade CVD |
| `ta_*` (3) | float32 | Trade Arrival |
| `ms_*` (7) | float32 | Microstructure |
| `rv_*` (6) | float32 | Realized Volatility |
| `sd_*` (8) | float32 | Sub-Bar Dynamics |
| `vp_*` (5) | float32 | Volume Profile |
| `iv_*` (13) | float32 | Options IV Surface |
| `gx_*` (4) | float32 | Dealer GEX |
| `dc_*` (4) | float32 | Daily Context |
| `wv_*` (14) | float32 | Wavelets |
| `cs_*` (33) | float32 | Candle Structure (includes 5s OHLCV `cs_open`…`cs_return`, ADR-069) |
| `vw_*` (5) | float32 | VWAP |
| `ca_*` (12) | float32 | Cross-Asset |
| `tc_*` (24) | float32 | Time Context |
| `ec_*` (4) | float32 | Economic Calendar |
| `hp_*` (3) | float32 | Hawkes Clustering |
| `ht_*` (5) | float32 | Higher Timeframe |
| `so_*` (6) | float32 | Stochastic %D + `so_sync_score_all` / `so_sync_score_9_60` |
| `vf_*` (20) | float32 | Closed 1-min / 5-min wick/body volume, delta, scaled delta; 1m POC + EMA (ADR-056) |
| `ps_*` (4) | float32 | 1m SAR trail + 5s pierce (`ps_lh_dist_sar`, `ps_flip`) (ADR-055) |
| `_meta_nq_contract` | string | Active NQ contract (e.g. `NQZ5`). Excluded from training. |
| `_meta_es_contract` | string | Active ES contract (e.g. `ESZ5`). Excluded from training. |

### 1.4 Trainer Column Exclusion Rules

`src/training/data_pipeline.py` excludes:
- All columns starting with `_` (metadata columns)
- The label column (computed at training time, never stored in Parquet)

### 1.5 Schema Compatibility Check

At serve time, `src/common/data_manifest.py` verifies that the live feature DataFrame matches the training-time schema:

```python
from src.common.data_manifest import assert_schema_compatible
assert_schema_compatible(train_schema, serve_df)
# Raises on: missing columns, extra columns, or dtype changes
```

`train_schema` is stored as `feature_schema.json` in each checkpoint directory.

### 1.6 Path-label parquet files (`parquet_pathlabels`)

Honest first-touch labels + dense targets from the 1s high/low path
([ADR-053](../decisions/ADR-053-vol-normalized-path-labels-and-encoder-ablation.md)).
**Not a feature cache** — join onto feature parquets on `ts`. Schema is the return of
`src.training.path_labels.label_session`. Built by
`scripts/features/build_path_labels.py --out …`.

```
$VTECH_DATA_DIR/sources/features/parquet_pathlabels/labels_YYYY-MM-DD.parquet
$VTECH_DATA_DIR/sources/features/parquet_pathlabels/_geography.csv
```

| Property | Value (canonical `vol_normalized` build, 2026-08-21) |
|----------|------------------------------------------------------|
| Files | **3,906** sessions (2011-01-10 → 2026-06-30) |
| Rows / session | 9,000 at `--entry-end 11:00` (08:30–10:59:59 CT, 1s); fewer on short days |
| Total rows | **35.1 M** |
| Index | none — `ts` is a column (`datetime64[ns, UTC]`) |
| Source path | `parquet_path1s/path_YYYY-MM-DD.parquet` |

| Column | Dtype | Meaning |
|--------|-------|---------|
| `ts` | datetime64[ns, UTC] | Entry time; join key |
| `label` | int32 | Sparse 3-class: `0=BIG_DOWN`, `1=NO_SIGNAL`, `2=BIG_UP` |
| `event_up` / `event_down` | int8 | First-touch on that side: `0=censored`, `1=target`, `2=stop` |
| `bucket_up` / `bucket_down` | int16 | 30 s bin of the touch (`0..19`); **20** = censored sentinel |
| `offset_up` / `offset_down` | int32 | Seconds to first touch; 600 if censored |
| `mfe_{up,down}_points` / `mae_{up,down}_points` | float64 | Max favourable / adverse excursion, points |
| `mfe_{up,down}_r` / `mae_{up,down}_r` | float64 | Same, divided by causal `R` |
| `target_points` / `stop_points` | float64 | Barriers actually applied this session (`target_r×R`, `stop_r×R`) |
| `r_value` | float64 | Causal R (median of prior 5 sessions' median 10-min range) |
| `stop_ticks` | float64 | `stop_points / 0.25` |
| `usable` | bool | `stop_ticks >= 8`. Gate for supervised `{0,1,2}` loss |
| `ambiguous` | bool | One 1s bar straddled both barriers |
| `truncated` | bool | Horizon ran off the end of the 1s grid |

Downstream **must** join on `ts` and respect `usable`. Hazard / quantile heads train on
every row (`src/models/heads.py`); the sparse `label` is supervision-gated.

Search fold cache (`labels.scheme: path_vol_normalized`) inner-joins these files onto
the 5s feature index (exact `ts` match → 1,800 rows/day in 08:30–11:00 CT). Feature
parquet currently starts 2025-04-07, so the trainable overlap with path labels is
~302 usable sessions through 2026-06-30.

### 1.7 Native-TF hive (`timeframe=/symbol=/year=/month=`)

15s 1-bar files are written ([ADR-075](../decisions/ADR-075-native-tf-15s-bar-features.md)).
5s research matrix uses the same hive ([ADR-079](../decisions/ADR-079-5s-hive-monthly-parts.md)).
1m files are written ([ADR-085](../decisions/ADR-085-native-tf-1m-features.md)).
5m files are written ([ADR-090](../decisions/ADR-090-native-tf-5m-features.md)).
1h / 1d HTF-context files are written ([ADR-091](../decisions/ADR-091-native-tf-1h-1d-htf-context.md)).

```
$VTECH_DATA_DIR/sources/features/timeframe=5s/symbol=NQ/year=YYYY/month=MM/features_5s_partN.parquet
$VTECH_DATA_DIR/sources/features/timeframe=15s/symbol=NQ/year=YYYY/month=MM/features_15s_partN.parquet
$VTECH_DATA_DIR/sources/features/timeframe=1m/symbol=NQ/year=YYYY/month=MM/features_1m_partN.parquet
$VTECH_DATA_DIR/sources/features/timeframe=5m/symbol=NQ/year=YYYY/month=MM/features_5m_partN.parquet
$VTECH_DATA_DIR/sources/features/timeframe=1h/symbol=NQ/year=YYYY/month=MM/features_1h_partN.parquet
$VTECH_DATA_DIR/sources/features/timeframe=1d/symbol=NQ/year=YYYY/month=MM/features_1d_partN.parquet
```

| Property | Value |
|----------|-------|
| Root | `sources/features/` (hive keys are siblings of leftover `parquet_5s/`) |
| Timeframes | `5s` (research matrix), `15s` (1-bar files), `1m` (ADR-085), `5m` (ADR-090), `1h`/`1d` HTF context (ADR-091) |
| Symbol | `NQ` (root, not contract) |
| Datetime | `bar_end_ts` (UTC). 15s closed-bar = left label + 15s; 1m = left label + 1m; 5m = left label + 5m; 1h = left label + 1h; 1d = next Chicago 17:00 |
| 5s features | Full catalog — [FEATURE_CATALOG.md](FEATURE_CATALOG.md); 08:00–11:00 CT |
| 15s features | 14 1-bar + 5 RTH/5-bar (ADR-083) + 4 `lg_*`/`st_*` target-geometry — [FEATURE_CATALOG_15s.md](FEATURE_CATALOG_15s.md) |
| 1m features | 71 cols `so_`/`cs_`/`ps_`/`vw_`/`rv_`/`vp_`/`ms_` (no options IV) — [FEATURE_CATALOG_1m.md](FEATURE_CATALOG_1m.md) |
| 5m features | 53 cols `cs_`/`vp_`/`vw_`/`ps_`/`zz_`/`vf_`/`rv_` (VWAP ±2σ; ATR(5) on the 5m bar) — [FEATURE_CATALOG_5m.md](FEATURE_CATALOG_5m.md) |
| 1h features | 45 cols `cs_`/`tr_`/`ps_`/`zz_`/`rv_`/`sr_` from `tbbo_1h` — [FEATURE_CATALOG_1h.md](FEATURE_CATALOG_1h.md) |
| 1d features | 45 cols `cs_`/`tr_`/`ps_`/`zz_`/`rv_`/`sr_` from Globex `tbbo_1d` — [FEATURE_CATALOG_1d.md](FEATURE_CATALOG_1d.md) |
| 15s / 1m / 5m / 1h / 1d coverage | Full Globex window (lead-month NQ), not 08:00–11:00 CT |
| Parts | `features_{tf}_partN.parquet`, cap ~100 MB / file |
| Hive month | UTC year/month of `bar_end_ts` |
| Path helper | `native_tf_dir` / `native_tf_part_path` / `native_tf_part_files` / `iter_hive_part_files` |

Do not scan `sources/features/` as one Hive dataset.

### 1.8 Label parquets (ADR-076)

| Store | Path | Columns |
|-------|------|---------|
| 5s 3-class TBM | `sources/labels/5s/<contract_id>/labels_5s_partN.parquet` | `ts`, `label` ∈ {0,1,2}, `close`, plus 10 path-quality aux (ADR-082 / ADR-107) |
| 15s dual-binary | `sources/labels/parquet_labels_15s/` (fallback `sources/features/parquet_labels_15s/`) | `ts`, `y_long`, `y_short`, plus 10 path-quality aux (ADR-082) |

5s sidecar `_contract.json`: scheme, move_threshold, mae_tolerance, horizon_s. Fold cache joins on `ts` (15s: `bar_end_ts` = label `ts`).

---

## 2. Manifest Sidecar Format

Every Parquet file has an accompanying `.manifest.json` written by `src/common/data_manifest.py` `write_manifest()`.

### 2.1 Example

```json
{
  "filename": "features_2025-10-15.parquet",
  "sha256": "a3b1c4d5e6f7...",
  "row_count": 1080,
  "col_count": 185,
  "size_bytes": 847320,
  "columns": {
    "bp_spread_mean": "float32",
    "bp_spread_max": "float32",
    "...(183 feature columns)...": "float32",
    "_meta_nq_contract": "object",
    "_meta_es_contract": "object"
  }
}
```

### 2.2 `diff_schemas()` Output

```python
# src/common/data_manifest.py
diff = diff_schemas(schema_a, schema_b)
# Returns: {"missing": [...], "extra": [...], "changed_dtype": {...}}
```

---

## 3. TimescaleDB Table Schemas

### 3.1 `databento.tbbo` — Trade + BBO

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `ts_event` | TIMESTAMPTZ | NOT NULL | Trade event timestamp (exchange, nanosecond precision) |
| `ts_recv` | TIMESTAMPTZ | NOT NULL | Gateway receive timestamp |
| `instrument_id` | INTEGER | NOT NULL | Databento instrument ID |
| `symbol` | TEXT | NOT NULL | Instrument symbol (e.g. `NQZ5`) |
| `price` | DOUBLE PRECISION | — | Trade price (USD) |
| `size` | INTEGER | — | Trade quantity (contracts) |
| `side` | CHAR(1) | — | `B` = buyer-initiated, `A` = seller-initiated |
| `sequence` | BIGINT | — | Exchange sequence number |
| `bid_px` | DOUBLE PRECISION | — | Best bid at time of trade |
| `ask_px` | DOUBLE PRECISION | — | Best ask at time of trade |
| `bid_sz` | INTEGER | — | Bid size |
| `ask_sz` | INTEGER | — | Ask size |
| `bid_ct` | INTEGER | — | Bid order count |
| `ask_ct` | INTEGER | — | Ask order count |
| `ts_in_delta` | INTEGER | — | Matching engine latency (ns) |
| `dataset` | TEXT | — | Default: `GLBX.MDP3` |

Primary key equivalent: `UNIQUE (instrument_id, ts_event, sequence)`

### 3.2 `databento.bbo_1s` — 1-Second BBO

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `ts_recv` | TIMESTAMPTZ | NOT NULL | Receive timestamp (partition key) |
| `instrument_id` | INTEGER | NOT NULL | Databento instrument ID |
| `symbol` | TEXT | NOT NULL | Instrument symbol |
| `price` | DOUBLE PRECISION | — | Last trade price in the 1-second window |
| `size` | INTEGER | — | Last trade size |
| `bid_px` | DOUBLE PRECISION | — | Best bid at interval close |
| `ask_px` | DOUBLE PRECISION | — | Best ask at interval close |
| `bid_sz` | INTEGER | — | — |
| `ask_sz` | INTEGER | — | — |
| `spread` | DOUBLE PRECISION | GENERATED | `ask_px − bid_px` (stored) |
| `dataset` | TEXT | — | Default: `GLBX.MDP3` |

Primary key equivalent: `UNIQUE (instrument_id, ts_recv)`

### 3.3 `databento.definitions` — Instrument Metadata

| Column | Type | Description |
|--------|------|-------------|
| `ts_event` | TIMESTAMPTZ | Definition snapshot timestamp |
| `instrument_id` | INTEGER | Databento instrument ID |
| `symbol` | TEXT | Symbol |
| `instrument_class` | CHAR(1) | `C`=Call, `P`=Put, `F`=Future |
| `strike_price` | DOUBLE PRECISION | Strike (options only) |
| `expiration` | TIMESTAMPTZ | Expiry date/time |
| `underlying` | TEXT | Root symbol (`NQ`, `ES`) |
| `exchange` | TEXT | — |
| `currency` | TEXT | Default `USD` |
| `min_price_increment` | DOUBLE PRECISION | Tick size |
| `multiplier` | DOUBLE PRECISION | Contract multiplier (NQ: $20) |
| `dataset` | TEXT | Default `GLBX.MDP3` |

### 3.4 `databento.statistics` — Daily Stats

| Column | Type | Description |
|--------|------|-------------|
| `ts_event` | TIMESTAMPTZ | Stat timestamp |
| `ts_recv` | TIMESTAMPTZ | Receive timestamp |
| `instrument_id` | INTEGER | — |
| `symbol` | TEXT | — |
| `stat_type` | SMALLINT | 1=Opening, 3=Settlement, 4=Session low, 5=Session high, 6=Cleared volume, 9=Open interest |
| `price` | DOUBLE PRECISION | Stat value (price fields) |
| `quantity` | BIGINT | Stat value (quantity fields) |
| `sequence` | BIGINT | — |
| `ts_ref` | TIMESTAMPTZ | Reference date for the stat |
| `update_action` | SMALLINT | 1=New, 2=Delete |
| `stat_flags` | SMALLINT | — |
| `dataset` | TEXT | Default `GLBX.MDP3` |

### 3.5 `databento.mbp_10` — Market-by-Price, Top 10 Levels (L2)

Databento `mbp-10` schema — one row per book-changing event, 10 price levels per side.
DBN `rtype` 10, 368 bytes on the wire. Rolling **1-month** window (L2 free tier).
See [DATABASE_SCHEMA.md §2.8](../architecture/DATABASE_SCHEMA.md) for DDL and design rationale.

| Column | Type | DBN wire type | Description |
|--------|------|---------------|-------------|
| `ts_recv` | TIMESTAMPTZ | `uint64` ns | Capture-server receive time. **Partition key**; the ordering Databento sorts MBP records by |
| `ts_event` | TIMESTAMPTZ | `uint64` ns | Matching-engine event time |
| `instrument_id` | INTEGER | `uint32` | Numeric instrument id |
| `symbol` | TEXT | — | Resolved raw symbol (e.g. `NQU6`) |
| `publisher_id` | SMALLINT | `uint16` | `GLBX.MDP3` = 1 |
| `rtype` | SMALLINT | `uint8` | Always 10 |
| `action` | CHAR(1) | `char` | `A`=Add `C`=Cancel `M`=Modify `R`=Clear `T`=Trade `F`=Fill `N`=None |
| `side` | CHAR(1) | `char` | `B`=Bid `A`=Ask `N`=None |
| `depth` | SMALLINT | `uint8` | 0–9; book level the update occurred at |
| `price` | DOUBLE PRECISION | `int64` 1e-9 | Price of the triggering order/trade |
| `size` | INTEGER | `uint32` | Quantity of the triggering order/trade |
| `flags` | SMALLINT | `uint8` | Bit field: 128=LAST 64=TOB 32=SNAPSHOT 16=MBP_AGG 8=BAD_TS_RECV 4=MAYBE_BAD_BOOK |
| `ts_in_delta` | INTEGER | `int32` ns | Matching-engine send latency before `ts_recv` |
| `sequence` | BIGINT | `uint32` | Venue sequence number (BIGINT — `uint32` overflows INTEGER) |
| `bid_px_00`…`bid_px_09` | DOUBLE PRECISION | `int64` 1e-9 | Bid price, level 0 (best) → 9. NULL if level absent |
| `ask_px_00`…`ask_px_09` | DOUBLE PRECISION | `int64` 1e-9 | Ask price, level 0 (best) → 9. NULL if level absent |
| `bid_sz_00`…`bid_sz_09` | INTEGER | `uint32` | Total resting bid quantity at level |
| `ask_sz_00`…`ask_sz_09` | INTEGER | `uint32` | Total resting ask quantity at level |
| `bid_ct_00`…`bid_ct_09` | INTEGER | `uint32` | Number of bid orders at level |
| `ask_ct_00`…`ask_ct_09` | INTEGER | `uint32` | Number of ask orders at level |
| `dataset` | TEXT | — | Default `GLBX.MDP3` |

75 columns total. Prices are USD (`int64` nanos ÷ 1e9) — lossless on the CME 0.25 grid.

**Timestamps are microsecond, not nanosecond.** `TIMESTAMPTZ` rounds half-up, costing
≤500 ns. Use `sequence` to recover exact event order; use the `.dbn.zst` files if you
need true nanoseconds.

```sql
-- top-of-book spread and depth imbalance over a window
SELECT ts_recv, symbol,
       ask_px_00 - bid_px_00                                  AS spread,
       (bid_sz_00 - ask_sz_00)::float / (bid_sz_00 + ask_sz_00) AS tob_imbalance,
       bid_sz_00+bid_sz_01+bid_sz_02 AS bid_depth3,
       ask_sz_00+ask_sz_01+ask_sz_02 AS ask_depth3
FROM databento.mbp_10
WHERE symbol = 'NQU6'
  AND ts_recv >= '2026-08-19 14:00Z' AND ts_recv < '2026-08-19 14:05Z'
  AND action = 'T'          -- trade events only
ORDER BY ts_recv, sequence;
```

---

## 4. `tbbo_1min` Continuous Aggregate Schema

| Column | Type | Description |
|--------|------|-------------|
| `bucket` | TIMESTAMPTZ | 1-minute time bucket |
| `instrument_id` | INTEGER | — |
| `symbol` | TEXT | — |
| `open` | DOUBLE PRECISION | First price in bucket |
| `high` | DOUBLE PRECISION | Max price |
| `low` | DOUBLE PRECISION | Min price |
| `close` | DOUBLE PRECISION | Last price |
| `volume` | BIGINT | Total volume |
| `buy_volume` | BIGINT | Buyer-initiated volume |
| `sell_volume` | BIGINT | Seller-initiated volume |
| `trade_count` | BIGINT | — |
| `bid_px` | DOUBLE PRECISION | Last bid |
| `ask_px` | DOUBLE PRECISION | Last ask |
| `bid_sz` | INTEGER | — |
| `ask_sz` | INTEGER | — |

---

## 5. Experiment Log — `_experiments.jsonl`

Location: `/opt/vtech/data/checkpoints/_experiments.jsonl`  
Format: JSON Lines (one JSON object per line, appended by `src/common/repro.py` `log_experiment()`)

### 5.1 Schema

| Field | Type | Description |
|-------|------|-------------|
| `ts` | ISO datetime | UTC timestamp of run |
| `git_sha` | string | Short HEAD SHA at training time |
| `seed` | int | Global seed used |
| `config_hash` | string | SHA-256 (12-char) of the config dict |
| `n_features` | int | Number of feature columns |
| `n_train_seq` | int | Number of training sequences |
| `test_accuracy` | float | Top-1 accuracy on test split |
| `test_f1_macro` | float | Macro F1 on test split |
| `training_seconds` | int | Wall time in seconds |

### 5.2 Read with pandas

```python
import pandas as pd
df = pd.read_json("/opt/vtech/data/checkpoints/_experiments.jsonl", lines=True)
df.sort_values("ts").tail(5)
```

---

## 6. Checkpoint Directory Structure

```
data/checkpoints/model_vNNN_name/
├── model.keras              ← Keras SavedModel (TF 2.15+)
├── features.json            ← Ordered list of feature column names
├── feature_schema.json      ← {col: dtype} dict for serve-time validation
└── training_result.json     ← {test_accuracy, test_f1_macro, epochs, ...}
```
