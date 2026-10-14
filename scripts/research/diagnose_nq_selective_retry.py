"""Read-only diagnosis of the failed experiment; writes new diagnosis artifacts.

Does not inspect the July 2026 reserve or select a replacement strategy.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from src.common.data_paths import resolve_data_path
from src.common.vm_config import get_vm_config
from scripts.research.train_nq_selective import book

RUN = Path('data/research/runs/nq-selective-intraday/20260924T033352Z/pass_1')
OUT = Path('data/research/diagnostics/nq-retry-20260924')


def coverage_day(info: dict) -> dict:
    """Measure stop-ceiling exclusions on the predeclared decision grid."""
    assert info['session'] < '2026-07-01'
    t = pd.read_parquet(resolve_data_path(info['path']), columns=['price'])
    bars = t.price.resample('1min', closed='left', label='right').ohlc()
    bars = bars.loc[bars.index <= t.index[-1]].copy()
    bars['close'] = bars.close.ffill()
    for c in ['open', 'high', 'low']:
        bars[c] = bars[c].fillna(bars.close)
    previous = bars.close.shift()
    tr = pd.concat([bars.high-bars.low, (bars.high-previous).abs(),
                    (bars.low-previous).abs()], axis=1).max(axis=1)
    risk = (np.ceil(2*tr.rolling(20, min_periods=20).mean()/.25)*.25).clip(lower=8)
    day = pd.Timestamp(info['session'], tz='America/Chicago')
    grid = pd.date_range(day+pd.Timedelta(hours=8, minutes=45),
                         day+pd.Timedelta(hours=14, minutes=25), freq='5min').tz_convert('UTC')
    r = risk.reindex(grid)
    return dict(session=info['session'], potential=len(grid), stop_over_60=int((r>60).sum()),
                missing_atr=int(r.isna().sum()), actual_eligible=info['decisions'],
                median_stop=float(r.median()))


def summary(g: pd.DataFrame) -> dict:
    return dict(n=len(g), mean_pred=float(g.predicted_r.mean()),
                mean_realized_r=float(g.net_r.mean()), net_usd=float(g.net_usd.sum()),
                correlation=float(g.predicted_r.corr(g.net_r)),
                short_fraction=float(g.side.eq(-1).mean()),
                stop_fraction=float(g.exit_reason.eq('stop').mean()))


def main() -> None:
    timer = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    f = pd.read_parquet(RUN/'training_matrix.parquet')
    y = pd.read_parquet(RUN/'candidate_outcomes.parquet')
    assert f.session.max() < '2026-07-01' and y.session.max() < '2026-07-01'
    model_root = RUN/'model_selective_ev'
    calibration, buckets, books = [], [], []
    for folder in sorted(model_root.glob('fold_*')):
        if not (folder/'model.pkl').exists():
            continue
        split = json.loads((folder/'split.json').read_text())
        meta = json.loads((folder/'model.json').read_text())
        with (folder/'model.pkl').open('rb') as h:
            model = pickle.load(h)
        for stage in ['train', 'inner', 'test']:
            low, high = split[stage]
            x = f.loc[f.session.between(low, high)]
            predictions = model.predict(x[meta['feature_columns']])
            scored = []
            for j, side in enumerate([-1, 1]):
                pred = pd.DataFrame({'ts':x.index, 'side':side, 'predicted_r':predictions[:, j]})
                scored.append(y[y.horizon.eq(meta['horizon']) & y.side.eq(side)
                                & y.session.between(low, high)].merge(pred, on=['ts','side']))
            scored = pd.concat(scored).sort_values(['ts','side'])
            chosen = scored.sort_values('predicted_r', ascending=False).drop_duplicates('ts')
            for population, g in [('all', scored), ('max_side', chosen),
                                  ('above_threshold', chosen[chosen.predicted_r>=meta['threshold']])]:
                calibration.append(dict(fold=folder.name, stage=stage, population=population, **summary(g)))
            trades, _ = book(scored, meta['threshold'])
            books.append(dict(fold=folder.name, stage=stage, **summary(trades)))
            scored['bucket'] = pd.cut(scored.predicted_r, [-np.inf,0,.05,.15,.25,np.inf]).astype(str)
            for key, g in scored.groupby('bucket'):
                buckets.append(dict(fold=folder.name, stage=stage, bucket=key, **summary(g)))
    pd.DataFrame(calibration).to_csv(OUT/'prediction_calibration.csv', index=False)
    pd.DataFrame(buckets).to_csv(OUT/'prediction_buckets.csv', index=False)
    pd.DataFrame(books).to_csv(OUT/'policy_train_inner_outer.csv', index=False)
    trades = pd.read_parquet(model_root/'trade_ledger.parquet')
    trades['hour_ct'] = trades.ts.dt.tz_convert('America/Chicago').dt.hour
    trades['month'] = trades.session.str[:7]
    for dimension in ['side','hour_ct','month','horizon','exit_reason']:
        trades.groupby(dimension).agg(n=('net_usd','size'), net_usd=('net_usd','sum'),
            mean_prediction=('predicted_r','mean'), mean_r=('net_r','mean'),
            stop_mean=('stop_points','mean')).to_csv(OUT/f'ledger_by_{dimension}.csv')
    y['duration_minutes'] = (y.exit_ts-y.entry_ts).dt.total_seconds()/60
    y['month'] = y.session.str[:7]
    y.groupby(['month','horizon','side']).agg(n=('net_r','size'), mean_r=('net_r','mean'),
        mean_minutes=('duration_minutes','mean'), target_fraction=('exit_reason',lambda x:x.eq('target').mean()),
        stop_fraction=('exit_reason',lambda x:x.eq('stop').mean())).to_csv(OUT/'population_by_month_horizon_side.csv')
    overlap = []
    for (h, side), g in y.groupby(['horizon','side']):
        g = g.sort_values('ts')
        g['prev_r'] = g.groupby('session').net_r.shift()
        g['previous_exit'] = g.groupby('session').exit_ts.shift()
        overlap.append(dict(horizon=h, side=side, rows=len(g), sessions=g.session.nunique(),
            lag1_outcome_correlation=g.net_r.corr(g.prev_r),
            adjacent_label_overlap_fraction=(g.ts<g.previous_exit).mean(),
            mean_label_minutes=g.duration_minutes.mean()))
    pd.DataFrame(overlap).to_csv(OUT/'label_dependence.csv', index=False)
    info = json.loads((RUN/'input_inventory.json').read_text())
    print('[████████░░░░░░░░░░░░] Calibration complete; measuring stop exclusions',flush=True)
    with ProcessPoolExecutor(max_workers=get_vm_config().workers_for(mem_per_worker_gb=2.0)) as pool:
        coverage = pd.DataFrame(pool.map(coverage_day, info))
    coverage.to_csv(OUT/'coverage_by_session.csv', index=False)
    coverage['month'] = coverage.session.str[:7]
    coverage.groupby('month').agg(sessions=('session','size'), potential=('potential','sum'),
        stop_over_60=('stop_over_60','sum'),missing_atr=('missing_atr','sum'),
        eligible=('actual_eligible','sum'), median_stop=('median_stop','median')).to_csv(OUT/'coverage_by_month.csv')
    print(f'[████████████████████] ✅ Done in {time.monotonic()-timer:.1f}s',flush=True)


if __name__ == '__main__':
    main()
