# Futures Catalogue — GLBX.MDP3 ohlcv-1s

> **Source**: Databento GLBX.MDP3, `ohlcv-1s` schema, continuous front-month contracts (`stype_in=continuous`)
> **DB table**: `databento.ohlcv_1s`
> **Total rows**: ~275.8 M (as of 2026-06-16)
> **Compressed size**: ~3.1 GB (23× compression ratio)
> **Ingested via**: `scripts/pipeline_ohlcv_all.py` + `scripts/ingest_dbn.py`
> **Definitions**: `databento.definitions` — 18,636 contract rows added 2026-06-17

---

## Data Schema

Each row is one **1-second OHLCV bar** for the front-month continuous contract:

| Column | Type | Description |
|---|---|---|
| `ts_event` | TIMESTAMPTZ | Bar open timestamp (exchange, UTC) — hypertable partition key |
| `instrument_id` | INTEGER | Databento numeric instrument ID |
| `symbol` | TEXT | Continuous contract symbol (e.g. `CL.c.0`) |
| `open` | DOUBLE PRECISION | Opening price of the 1-second bar |
| `high` | DOUBLE PRECISION | High price |
| `low` | DOUBLE PRECISION | Low price |
| `close` | DOUBLE PRECISION | Closing price |
| `volume` | BIGINT | Traded volume (contracts) |
| `dataset` | TEXT | Always `GLBX.MDP3` |

Prices are **unadjusted** (no back-adjustment across rolls). Hypertable chunk interval: **1 month**.

---

## Product Catalogue

35 products across 7 sectors. Sorted by sector, then ticker.

### Equity Index (5)

| Ticker | Symbol | Name | Exchange | Data Range | Notes |
|---|---|---|---|---|---|
| `ES` | `ES.c.0` | E-Mini S&P 500 | CME | 2010-06-06 → 2026-06-16 | |
| `NQ` | `NQ.c.0` | E-Mini Nasdaq 100 | CME | 2025-04-06 → 2026-04-05 | tbbo schema; ohlcv_1s available but separate |
| `RTY` | `RTY.c.0` | E-Mini Russell 2000 | CME | 2017-07-01 → 2026-06-03 | Launched Jul 2017 |
| `YM` | `YM.c.0` | E-Mini Dow Jones | CBOT | 2010-06-06 → 2026-06-03 | |
| `ZB`¹ | — | — | — | — | See Rates |

### Energy (6)

| Ticker | Symbol | Name | Exchange | Data Range | Notes |
|---|---|---|---|---|---|
| `CL` | `CL.c.0` | Crude Oil WTI | NYMEX | 2010-06-06 → 2026-06-15 | Highest volume energy futures |
| `HO` | `HO.c.0` | Heating Oil (NY Harbor ULSD) | NYMEX | 2010-06-06 → 2026-06-03 | |
| `NG` | `NG.c.0` | Natural Gas Henry Hub | NYMEX | 2010-06-06 → 2026-06-03 | |
| `QG` | `QG.c.0` | E-mini Natural Gas | NYMEX | 2010-06-06 → 2026-06-15 | 2,500 MMBtu (1/4 of NG) |
| `QM` | `QM.c.0` | E-mini Crude Oil | NYMEX | 2010-06-06 → 2026-06-15 | 500 bbl (1/2 of CL) |
| `RB` | `RB.c.0` | RBOB Gasoline | NYMEX | 2010-06-06 → 2026-06-03 | |

### Metals (4)

| Ticker | Symbol | Name | Exchange | Data Range | Notes |
|---|---|---|---|---|---|
| `GC` | `GC.c.0` | Gold | COMEX | 2010-06-06 → 2026-06-03 | |
| `HG` | `HG.c.0` | Copper | COMEX | 2010-06-06 → 2026-06-03 | |
| `SI` | `SI.c.0` | Silver | COMEX | 2010-06-06 → 2026-06-03 | |

### FX (5)

| Ticker | Symbol | Name | Exchange | Data Range | Notes |
|---|---|---|---|---|---|
| `6A` | `6A.c.0` | Australian Dollar | CME | 2010-06-06 → 2026-06-15 | 100,000 AUD per contract |
| `6B` | `6B.c.0` | British Pound | CME | 2010-06-06 → 2026-06-15 | 62,500 GBP per contract |
| `6C` | `6C.c.0` | Canadian Dollar | CME | 2010-06-06 → 2026-06-15 | 100,000 CAD per contract |
| `6E` | `6E.c.0` | Euro FX | CME | 2010-06-06 → 2026-06-03 | 125,000 EUR per contract |
| `6J` | `6J.c.0` | Japanese Yen | CME | 2010-06-06 → 2026-06-03 | 12,500,000 JPY per contract |

### Rates (8)

| Ticker | Symbol | Name | Exchange | Data Range | Notes |
|---|---|---|---|---|---|
| `SR3` | `SR3.c.0` | 3-Month SOFR | CME | 2018-05-07 → 2026-06-03 | Launched May 2018 |
| `TN` | `TN.c.0` | Ultra 10Y T-Note | CBOT | 2016-01-11 → 2026-06-15 | Launched Jan 2016 |
| `UB` | `UB.c.0` | Ultra T-Bond (30Y+) | CBOT | 2010-06-06 → 2026-06-15 | |
| `ZB` | `ZB.c.0` | 30Y T-Bond | CBOT | 2010-06-06 → 2026-06-15 | |
| `ZF` | `ZF.c.0` | 5Y T-Note | CBOT | 2010-06-06 → 2026-06-15 | |
| `ZN` | `ZN.c.0` | 10Y T-Note | CBOT | 2010-06-06 → 2026-06-03 | |
| `ZT` | `ZT.c.0` | 2Y T-Note | CBOT | 2010-06-06 → 2026-06-15 | |

### Agricultural (8)

| Ticker | Symbol | Name | Exchange | Data Range | Notes |
|---|---|---|---|---|---|
| `XC` | `XC.c.0` | Mini Corn | CBOT | 2010-06-06 → 2026-06-15 | 1,000 bu (1/5 of ZC) |
| `XK` | `XK.c.0` | Mini Soybeans | CBOT | 2010-06-06 → 2026-06-15 | 1,000 bu (1/5 of ZS) |
| `XW` | `XW.c.0` | Mini SRW Wheat | CBOT | 2010-06-06 → 2026-06-15 | 1,000 bu (1/5 of ZW) |
| `ZC` | `ZC.c.0` | Corn | CBOT | 2010-06-06 → 2026-06-03 | 5,000 bu |
| `ZS` | `ZS.c.0` | Soybeans | CBOT | 2010-06-06 → 2026-06-03 | 5,000 bu |
| `ZW` | `ZW.c.0` | Chicago SRW Wheat | CBOT | 2010-06-06 → 2026-06-15 | 5,000 bu |

### Livestock (2)

| Ticker | Symbol | Name | Exchange | Data Range | Notes |
|---|---|---|---|---|---|
| `HE` | `HE.c.0` | Lean Hog | CME | 2010-06-06 → 2026-06-15 | 40,000 lbs |
| `LE` | `LE.c.0` | Live Cattle | CME | 2010-06-06 → 2026-06-15 | 40,000 lbs |

### Crypto (2)

| Ticker | Symbol | Name | Exchange | Data Range | Notes |
|---|---|---|---|---|---|
| `BTC` | `BTC.c.0` | Bitcoin | CME | 2017-12-18 → 2026-06-15 | 5 BTC per contract; launched Dec 2017 |
| `ETH` | `ETH.c.0` | Ether | CME | 2021-02-08 → 2026-06-15 | 50 ETH per contract; launched Feb 2021 |

---

## Coverage Summary

| Sector | Products | Earliest Data |
|---|---|---|
| Equity Index | ES, NQ, RTY, YM | 2010-06-06 (NQ: 2025-04-06 tbbo only) |
| Energy | CL, HO, NG, QG, QM, RB | 2010-06-06 |
| Metals | GC, HG, SI | 2010-06-06 |
| FX | 6A, 6B, 6C, 6E, 6J | 2010-06-06 |
| Rates | SR3, TN, UB, ZB, ZF, ZN, ZT | 2010-06-06 (SR3: 2018, TN: 2016) |
| Agricultural | XC, XK, XW, ZC, ZS, ZW | 2010-06-06 |
| Livestock | HE, LE | 2010-06-06 |
| Crypto | BTC, ETH | 2017-12 / 2021-02 |

Most products: **~16 years** of 1-second bars (2010-06-06 → present), ~193 monthly files each.

---

## Ingest Details

- **Ingest script**: `scripts/pipeline_ohlcv_all.py` (download + ingest + compress in one pass)
- **Ingest worker**: `scripts/ingest_dbn.py --table ohlcv_1s`, ProcessPoolExecutor, COPY pattern
- **Checkpoints**: `data/checkpoints/ingest_ohlcv_1s_{ticker}.json` — idempotent, safe to re-run
- **Raw files**: `data/raw/backfill/{ticker}_ohlcv_1s/` — one subdirectory per batch job ID
- **Definitions**: `data/raw/backfill/definitions/` — 34 × `{TICKER}_FUT_definition.dbn.zst`, 60-day window, ingested 2026-06-17 via `scripts/download_ingest_definitions.py`
- **Compression**: TimescaleDB auto-compresses chunks older than 30 days (~23× ratio)
