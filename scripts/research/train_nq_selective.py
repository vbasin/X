#!/usr/bin/env python3
"""Preregistered NQ experiment. New artifacts only; never trades or promotes.

Run from repository root: venv/bin/python -m scripts.research.train_nq_selective
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.common.data_paths import logical_path
from src.common.cme_holidays import is_trading_day
from src.common.run_layout import new_run_ts, run_dir
from src.common.vm_config import get_vm_config
from src.evaluation.economic import (Leg, LegRole, OrderType, Trade,
    compute_trade_ledger, load_assumptions)
from src.experiment.manifest import CodeIdentity, DatasetRef, RunManifest, ValidationDesign
from src.features.tape import list_tape_files

OBJECTIVE = 'nq-selective-intraday'
SEED = 20260924
CT = 'America/Chicago'
ASSUMPTIONS = replace(load_assumptions('config/execution_costs.yaml','NQ'),
                      slippage_points_per_side=.25,
                      fill_model='print-associated executable BBO, 250ms latency, adverse slippage, market exits')
COMMISSION = ASSUMPTIONS.commission_per_trade
TEMPLATES = [(15, 1.5), (30, 2.0)]
THRESHOLDS = [.05, .15, .25]
OUTERS = [('2026-01-01', '2026-03-01'), ('2026-03-01', '2026-05-01'),
          ('2026-05-01', '2026-07-01')]
MODEL_PARAMS = dict(n_estimators=250, max_depth=2, learning_rate=.03,
                    min_child_weight=100, reg_lambda=20, subsample=.8,
                    colsample_bytree=.8, random_state=SEED, n_jobs=1,
                    tree_method='hist', objective='reg:squarederror')
SOURCES = ['scripts/research/train_nq_selective.py', 'src/features/nq_selective.py',
           'src/backtest/nq_selective.py', 'src/common/causal.py',
           'src/evaluation/economic.py', 'config/execution_costs.yaml', 'src/common/cme_holidays.py', 'config/objectives/nq-selective-intraday.yaml',
           'docs/research/NQ_SELECTIVE_PROTOCOL.md']


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + '\n')


def progress(stage: str, fraction: float, start: float) -> None:
    n = int(20 * fraction)
    print(f"[{'█'*n}{'░'*(20-n)}] {stage} ({time.monotonic()-start:.0f}s)", flush=True)


def build_day(path_str: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from src.features.nq_selective import build_features
    from src.backtest.nq_selective import simulate_candidates
    path = Path(path_str)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    tape = pd.read_parquet(path)
    feat = build_features(tape)
    if 'stop_points' not in feat:
        feat['stop_points'] = pd.Series(index=feat.index, dtype=float)
    session = path.stem.replace('tape_', '')
    feat['session'] = session
    outcomes = []
    for horizon, target in TEMPLATES:
        y = simulate_candidates(tape, feat, horizon_minutes=horizon, target_r=target,commission=COMMISSION)
        y['horizon'] = horizon
        y['session'] = session
        outcomes.append(y)
    y = pd.concat(outcomes, ignore_index=True)
    info = dict(session=session, path=logical_path(path), sha256=digest,
                n_prints=len(tape), decisions=len(feat), invalid=int((~y.valid).sum()))
    return feat, y, info


def build_data(files: list[Path], output: Path, start: float) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    workers = get_vm_config().workers_for(mem_per_worker_gb=2.0)
    frames, labels, inventory = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(build_day, str(p)): p for p in files}
        for i, future in enumerate(as_completed(jobs), 1):
            f, y, info = future.result()
            frames.append(f)
            labels.append(y)
            inventory.append(info)
            if i % 20 == 0 or i == len(files):
                progress(f'Build {i}/{len(files)} sessions', i/len(files), start)
    feat = pd.concat(frames).sort_index()
    y = pd.concat(labels, ignore_index=True).sort_values(['ts','horizon','side'])
    output.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(output / 'training_matrix.parquet')
    y.to_parquet(output / 'candidate_outcomes.parquet', index=False)
    inventory.sort(key=lambda d:d['session'])
    dump(output / 'input_inventory.json', inventory)
    return feat, y, inventory


def book(scored: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[dict]]:
    """Select before consulting outcomes; no overlapping fills or hidden censor filter."""
    if scored.empty:
        return scored.copy(), []
    choices = scored.sort_values(['ts', 'predicted_r', 'side'], ascending=[True,False,False])
    choices = choices.drop_duplicates('ts').sort_values('ts')
    trades, unresolved = [], []
    session, busy, count, losses, realized = None, None, 0, 0, 0.0
    for row in choices.itertuples(index=False):
        if row.session != session:
            session, busy, count, losses, realized = row.session, None, 0, 0, 0.0
        if busy is not None and row.ts <= busy:
            continue
        if count >= 4 or losses >= 2 or realized <= -2 or row.predicted_r < threshold:
            continue
        if not np.isfinite(row.predicted_r):
            continue
        if not row.valid:
            unresolved.append(dict(session=session, ts=str(row.ts), reason=row.invalid_reason))
            # Fail closed for this session; never use an incomplete path to keep trading.
            count = 4
            continue
        trades.append(row._asdict())
        busy = row.exit_ts
        count += 1
        losses += row.net_usd < 0
        realized += row.net_r
    return pd.DataFrame(trades, columns=scored.columns), unresolved


def summarize(trades: pd.DataFrame, sessions: list[str], bootstrap: bool = False) -> tuple[dict, pd.DataFrame]:
    daily = pd.DataFrame(index=pd.Index(sessions, name='session'))
    if len(trades):
        daily['net_usd'] = trades.groupby('session').net_usd.sum()
    else:
        daily['net_usd'] = 0.
    daily['net_usd'] = daily.net_usd.fillna(0.)
    daily['equity_usd'] = daily.net_usd.cumsum()
    daily['drawdown_usd'] = daily.equity_usd - daily.equity_usd.cummax().clip(lower=0.)
    net = trades.net_usd.to_numpy(float) if len(trades) else np.array([])
    winners, losers = net[net>0], net[net<0]
    equity = np.r_[0., np.cumsum(net)]
    dd = float(np.max(np.maximum.accumulate(equity)-equity))
    streak = worst = 0
    for value in net:
        streak = streak+1 if value<0 else 0
        worst = max(worst, streak)
    # Closed-equity peak to liquidation trough proxy; excludes unrealized equity peaks.
    if len(trades):
        mtm = equity[:-1] - trades.mae_points.to_numpy(float)*20 - COMMISSION - 5.0
        peak_before = np.maximum.accumulate(equity)[:-1]
        mark_dd = float(np.max(peak_before-mtm))
    else:
        mark_dd = 0.
    stats = dict(n_trades=len(net), n_sessions=len(sessions), net_usd=float(net.sum()),
                 gross_profit_usd=float(winners.sum()), gross_loss_usd=float(-losers.sum()),
                 loss_count=len(losers), win_rate=float((net>0).mean()) if len(net) else 0.,
                 profit_factor=float(winners.sum()/-losers.sum()) if len(losers) else None,
                 avg_trade_usd=float(net.mean()) if len(net) else 0.,
                 max_drawdown_usd=dd, closed_peak_to_intratrade_trough_proxy_usd=mark_dd,
                 max_consecutive_losses=worst,
                 mean_daily_usd=float(daily.net_usd.mean()),
                 worst_day_usd=float(daily.net_usd.min()),
                 positive_days=int((daily.net_usd>0).sum()),
                 worst_trade_usd=float(net.min()) if len(net) else 0.)
    if bootstrap:
        rng = np.random.default_rng(SEED)
        values = daily.net_usd.to_numpy()
        if len(values):
            # Circular moving block resampling preserves local five-session dependence.
            idx = (rng.integers(0,len(values),(5000,(len(values)+4)//5,1)) + np.arange(5)) % len(values)
            means = values[idx.reshape(5000,-1)[:,:len(values)]].mean(axis=1)
            stats['daily_mean_95ci'] = np.quantile(means,[.025,.975]).tolist()
        else:
            stats['daily_mean_95ci'] = [0.,0.]
    return stats, daily


def feature_views(feat: pd.DataFrame) -> dict[str,list[str]]:
    full = [c for c in feat if c.startswith('nq_')]
    flow_terms = ('delta','imbalance','ofi','flow','volume','spread')
    price = [c for c in full if not any(t in c for t in flow_terms)]
    return {'xgb_full':full, 'xgb_price':price, 'ridge_price':price}


def fit_one(task: tuple) -> dict:
    name, horizon, cols, feat, outcomes, train_sessions, inner_sessions, test_sessions = task
    data = outcomes[outcomes.horizon.eq(horizon)]
    train_y = data[data.session.isin(train_sessions) & data.valid].pivot(index='ts',columns='side',values='net_r').dropna()
    x = feat.loc[train_y.index, cols].replace([np.inf,-np.inf], np.nan)
    if name.startswith('ridge'):
        model = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                              StandardScaler(),Ridge(alpha=100.))
    else:
        model = XGBRegressor(**MODEL_PARAMS)
    model.fit(x, train_y[[-1,1]].to_numpy())
    pieces = {}
    for stage, sessions in [('inner',inner_sessions),('test',test_sessions)]:
        xf = feat[feat.session.isin(sessions)]
        pred = model.predict(xf[cols].replace([np.inf,-np.inf],np.nan))
        scored = []
        for j, side in enumerate([-1,1]):
            p = pd.DataFrame({'ts':xf.index,'side':side,'predicted_r':pred[:,j]})
            scored.append(data[data.session.isin(sessions) & data.side.eq(side)].merge(p,on=['ts','side'],how='left',validate='one_to_one'))
        pieces[stage] = pd.concat(scored, ignore_index=True)
    return dict(name=name,horizon=horizon,cols=cols,model=model,**pieces)


def run_fold(feat: pd.DataFrame, outcomes: pd.DataFrame, all_sessions: list[str],
             start: str, end: str, output: Path, timer: float) -> tuple[pd.DataFrame,dict,dict]:
    inner_start = str((pd.Timestamp(start)-pd.DateOffset(months=2)).date())
    train = [s for s in all_sessions if s<inner_start][:-1]
    inner = [s for s in all_sessions if inner_start<=s<start][:-1]
    test = [s for s in all_sessions if start<=s<end]
    if not train or not inner or not test:
        raise ValueError('empty temporal split')
    train_end = outcomes.loc[outcomes.session.isin(train)&outcomes.valid,'exit_ts'].max()
    inner_first = feat[feat.session.isin(inner)].index.min()
    inner_end = outcomes.loc[outcomes.session.isin(inner)&outcomes.valid,'exit_ts'].max()
    test_first = feat[feat.session.isin(test)].index.min()
    assert train_end < inner_first and inner_end < test_first
    split = dict(train=[train[0],train[-1]],inner=[inner[0],inner[-1]],test=[test[0],test[-1]],
                 max_train_label_end=str(train_end), first_inner_decision=str(inner_first),
                 max_inner_label_end=str(inner_end), first_test_decision=str(test_first))
    output.mkdir(parents=True, exist_ok=True)
    dump(output/'split.json',split)
    tasks = [(name,horizon,cols,feat,outcomes,train,inner,test)
             for name,cols in feature_views(feat).items() for horizon,_ in TEMPLATES]
    trials, results = [], []
    with ThreadPoolExecutor(max_workers=get_vm_config().workers_for(mem_per_worker_gb=2.0)) as pool:
        for result in pool.map(fit_one,tasks):
            results.append(result)
            for threshold in THRESHOLDS:
                trades, missing = book(result['inner'],threshold)
                stats, daily = summarize(trades,inner)
                score = float(daily.net_usd.mean()-.5*daily.net_usd.std(ddof=1)/np.sqrt(len(daily)))
                eligible = (stats['n_trades']>=30 and (stats['profit_factor'] or 0)>=1.10 and not missing)
                trials.append(dict(name=result['name'],horizon=result['horizon'],threshold=threshold,
                                   score=score,eligible=eligible,unresolved=len(missing),**stats))
    dump(output/'selection_trials.json',trials)
    eligible = [t for t in trials if t['eligible'] and t['score']>0]
    if not eligible:
        split['selected'] = 'cash'
        empty = outcomes.iloc[:0].copy()
        empty['predicted_r'] = pd.Series(dtype=float)
        dump(output/'selection.json',split)
        return empty, split, {'results':results,'selected':None,'test_sessions':test}
    best = max(eligible,key=lambda t:t['score'])
    selected = next(r for r in results if (r['name'],r['horizon'])==(best['name'],best['horizon']))
    trades, missing = book(selected['test'],best['threshold'])
    trades['fold'] = start
    split.update(selected={k:best[k] for k in ['name','horizon','threshold']},unresolved=missing)
    selected['test'].to_parquet(output/'oof_predictions.parquet',index=False)
    trades.to_parquet(output/'trade_ledger.parquet',index=False)
    with (output/'model.pkl').open('wb') as handle:
        pickle.dump(selected['model'],handle)
    dump(output/'model.json',dict(model_type=selected['name'],horizon=selected['horizon'],
         feature_columns=selected['cols'],target='[-1,+1] net_r',threshold=best['threshold'],
         hyperparameters=MODEL_PARAMS if selected['name'].startswith('xgb') else {'alpha':100.},
         training_cutoff=str(train_end),status='EXPERIMENTAL'))
    if hasattr(selected['model'],'feature_importances_'):
        pd.DataFrame({'feature':selected['cols'],'importance':selected['model'].feature_importances_}).sort_values('importance',ascending=False).to_csv(output/'feature_importance.csv',index=False)
    dump(output/'selection.json',split)
    progress(f'Frozen OOF {start}: {best["name"]}/{best["horizon"]}m',.7,timer)
    return trades,split,dict(results=results,selected=selected,threshold=best['threshold'],test_sessions=test)


def verify_economics(trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    assumptions = load_assumptions('config/execution_costs.yaml','NQ')
    resolved = [Trade(str(i),int(r.side),
        Leg(OrderType.MARKET,resolved_fill_price=r.entry_price,fill_source='nq_print_quote_latency_slippage',ts=str(r.entry_ts)),
        Leg(OrderType.MARKET,role=LegRole.EXIT_TIME,resolved_fill_price=r.exit_price,
            fill_source='nq_print_quote_latency_slippage',ts=str(r.exit_ts))) for i,r in enumerate(trades.itertuples())]
    ledger = compute_trade_ledger(resolved,assumptions)
    np.testing.assert_allclose([r.net_dollars for r in ledger.trades],trades.net_usd,rtol=0,atol=1e-7)


def baselines(feat: pd.DataFrame, outcomes: pd.DataFrame, sessions: list[str], output: Path) -> dict:
    """Fixed comparators; their outcomes never select the adaptive policy."""
    f = feat[feat.session.isin(sessions)]
    directions = {
        'trend_15m': np.sign(f.nq_return_15m),
        'fade_vwap_1R': -np.sign(f.nq_vwap_distance).where(f.nq_vwap_distance.abs()>=1,0),
        'opening_breakout': pd.Series(np.where(f.nq_opening_high_distance>0,1,
                                     np.where(f.nq_opening_low_distance<0,-1,0)),index=f.index),
        'long_only':pd.Series(1,index=f.index),
    }
    results = {'cash':summarize(outcomes.iloc[:0],sessions)[0]}
    for name,direction in directions.items():
        prediction = pd.DataFrame({'ts':f.index,'direction':direction.to_numpy()})
        scored = outcomes[outcomes.horizon.eq(30)&outcomes.session.isin(sessions)].merge(prediction,on='ts',validate='many_to_one')
        scored['predicted_r'] = np.where(scored.side.eq(scored.direction),1.,-1.)
        trades,missing = book(scored,.5)
        results[name] = dict(summarize(trades,sessions)[0],unresolved=len(missing))
        trades.to_parquet(output/f'baseline_{name}.parquet',index=False)
    dump(output/'baselines.json',results)
    return results


def stress_day(task: tuple) -> pd.DataFrame:
    from src.backtest.nq_selective import simulate_candidates
    path, decision_rows, horizon, slip, latency = task
    tape = pd.read_parquet(path)
    target = dict(TEMPLATES)[horizon]
    out = simulate_candidates(tape,decision_rows.set_index('ts'),horizon_minutes=horizon,
                              target_r=target,slippage_points=slip,latency_ms=latency,commission=COMMISSION)
    out['session'] = Path(path).stem.replace('tape_','')
    out['horizon'] = horizon
    return out


def stress_fold(state: dict, files: list[Path], slip: float, latency: int) -> tuple[pd.DataFrame,list[dict]]:
    selected = state['selected']
    if selected is None:
        return pd.DataFrame(),[]
    scored = selected['test']
    by_session = {s:g for s,g in scored.groupby('session')}
    tasks = [(str(p),by_session[p.stem[5:]][['ts','stop_points']].drop_duplicates('ts'),
              selected['horizon'],slip,latency) for p in files if p.stem[5:] in by_session]
    with ProcessPoolExecutor(max_workers=get_vm_config().workers_for(mem_per_worker_gb=2.0)) as pool:
        outcomes = pd.concat(pool.map(stress_day,tasks),ignore_index=True)
    predictions = scored[['ts','side','predicted_r']]
    rescored = outcomes.merge(predictions,on=['ts','side'],validate='one_to_one')
    return book(rescored,state['threshold'])


def report(output: Path, stats: dict, daily: pd.DataFrame, folds: list[dict],
           stresses: dict, gate: dict, timer: float) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    dump(output/'metrics.json',stats)
    dump(output/'fold_metrics.json',folds)
    dump(output/'cost_sensitivity.json',stresses)
    dump(output/'acceptance.json',gate)
    daily.to_csv(output/'daily_equity.csv')
    fig,axes = plt.subplots(2,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    dates = pd.to_datetime(daily.index)
    axes[0].plot(dates,daily.equity_usd,color='#246a73')
    axes[0].set(ylabel='Net P&L / one NQ ($)',title='Chronological OOF simulation — bid/ask, latency, commission, slippage')
    axes[1].fill_between(dates,daily.drawdown_usd,0,color='#bd4b4b')
    axes[1].set(ylabel='Daily drawdown ($)')
    fig.tight_layout()
    fig.savefig(output/'equity.png',dpi=150)
    plt.close(fig)
    rows = '\n'.join(f'| {k} | {v} |' for k,v in stats.items())
    fold_rows = '\n'.join(f'| {r["test"][0]}–{r["test"][1]} | {r["metrics"]["n_trades"]} | ${r["metrics"]["net_usd"]:,.2f} | {r["selected"]} |' for r in folds)
    stress_rows = '\n'.join(f'| {k} | ${v["net_usd"]:,.2f} | {v["n_trades"]} | {v["unresolved"]} |' for k,v in stresses.items())
    text = f'''# NQ selective intraday — physical research evidence

**Verdict: {gate['verdict']}**. This is historical simulation, not proof of future profit.
The July–September reserve is {gate['holdout_status']}.

## Measured development OOF results

One NQ contract; $20/point, actual bid/ask, $3.36 round-trip commission,
0.25 point adverse slippage per side and 250ms latency. No compounding.
Returns include zero-trade observed sessions. Profit factor uses net winning/losing
trades. Drawdown is reported both on closed trades and a closed-peak-to-intratrade-trough
liquidation proxy (not a bound on full marked-to-market drawdown); neither is a live account guarantee.

| Metric | Value |
|---|---:|
{rows}

## Chronological folds

Each model and threshold is selected using only the preceding inner window,
with an entire observed session purged at both boundaries. Label-end assertions
are stored in each split. Frozen models are not refitted on OOF.

| OOF period | Trades | Net P&L | Past-selected policy |
|---|---:|---:|---|
{fold_rows}

## Execution stress — full path resimulation with frozen predictions/policy

| Scenario | Net P&L | Trades | Unresolved signals |
|---|---:|---:|---:|
{stress_rows}

## Design and evidence

See the preregistered protocol snapshot and config in this run. Raw tape SHA256
hashes, source code snapshots, fitted estimators, all inner selection trials,
OOF predictions and transaction ledgers accompany this report. Selection searches
6 candidates × 3 thresholds on past-only data; no claim of global optimality.
Five-session moving-block bootstrap uses 5000 replicates. The development
interval does not correct for other historical research conducted by this user.

The new raw-tape features bypass a verified full-session normalization leak in
legacy es_cvd/es_delta/es_ofi features. No final-session maxima, future bars,
future normalizers or forward upstream predictions enter these new features.

## Practical limitations

Print-associated quotes are not a complete quote stream and do not establish
queue priority, actual latency or market impact. Scheduled news filters were not
retrospectively fitted. Censored paths and source coverage are disclosed. Account
capital, broker-specific fees and acceptable dollar drawdown were unspecified;
there is no percentage return, margin sufficiency or position-sizing claim.
The reserve is untouched by THIS experiment until allowed, but its possible use
in previous repository research prevents claiming a historically virgin sample.

Nothing was deployed, promoted, or connected to order entry. A failed gate means
cash is the supported operational choice. The candidate's fitted model files are
research artifacts, not a profitable live trading recommendation.

![OOF equity and drawdown](equity.png)

Completed in {time.monotonic()-timer:.0f} seconds.
'''
    (output/'summary.md').write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',type=Path,help='Resume this same protocol after interruption; reuse development cache only')
    args = parser.parse_args()
    timer = time.monotonic()
    root = args.resume or run_dir(OBJECTIVE,new_run_ts())
    root.mkdir(parents=True,exist_ok=True)
    model_root = root/'pass_1'/'model_selective_ev'
    model_root.mkdir(parents=True,exist_ok=True)
    print(f'RUN_DIR={root}',flush=True)
    files = [p for p in list_tape_files() if is_trading_day(pd.Timestamp(p.stem[5:]).date())
             and p.stem[5:] != '2025-07-03']
    dev_files = [p for p in files if p.stem[5:]<'2026-07-01']
    held_files = [p for p in files if '2026-07-01'<=p.stem[5:]<='2026-09-22']
    config = dict(seed=SEED,templates=TEMPLATES,thresholds=THRESHOLDS,outer_folds=OUTERS,
                  holdout=['2026-07-01','2026-09-22'],xgb=MODEL_PARAMS,ridge_alpha=100.,
                  max_entries=4,max_losing_trades=2,daily_loss_r=2.,
                  base_slippage_points=.25,latency_ms=250,commission=COMMISSION,
                  candidate_views=['xgb_full','xgb_price','ridge_price'])
    config['calendar'] = 'src.common.cme_holidays regular days; also exclude scheduled 2025-07-03 early close'
    if args.resume and (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text()) != json.loads(json.dumps(config)):
        raise ValueError('Resume refuses changed protocol; create a new run')
    dump(root/'protocol.json',config)
    code_dir = root/'code'
    code_dir.mkdir(exist_ok=True)
    hashes = {}
    for source in SOURCES:
        p = Path(source)
        data = p.read_bytes()
        hashes[source] = hashlib.sha256(data).hexdigest()
        target = code_dir/source
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=data:
            raise ValueError(f'Resume refuses changed source {source}; create a new run')
        if not target.exists():
            target.write_bytes(data)
    dump(root/'source_hashes.json',hashes)
    cache = root/'pass_1'
    if (cache/'training_matrix.parquet').exists() and (cache/'input_inventory.json').exists():
        feat = pd.read_parquet(cache/'training_matrix.parquet')
        outcomes = pd.read_parquet(cache/'candidate_outcomes.parquet')
        inventory = json.loads((cache/'input_inventory.json').read_text())
        current = {logical_path(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dev_files}
        if current != {r['path']:r['sha256'] for r in inventory}:
            raise ValueError('Resume refuses changed input bytes')
    else:
        feat,outcomes,inventory = build_data(dev_files,cache,timer)
    sessions = [p.stem[5:] for p in dev_files]
    dump(model_root/'data_quality.json',dict(sessions=len(sessions),n_decisions=len(feat),
         n_outcomes=len(outcomes),invalid=int((~outcomes.valid).sum()),
         invalid_reasons=outcomes.loc[~outcomes.valid,'invalid_reason'].value_counts().to_dict()))
    all_trades, fold_records, states = [], [], []
    for start,end in OUTERS:
        trades,split,state = run_fold(feat,outcomes,sessions,start,end,model_root/f'fold_{start}',timer)
        verify_economics(trades)
        stats,_ = summarize(trades,state['test_sessions'])
        split['metrics'] = stats
        fold_records.append(split)
        states.append(state)
        all_trades.append(trades)
    trades = pd.concat(all_trades,ignore_index=True).sort_values('ts')
    test_sessions = [s for s in sessions if '2026-01-01'<=s<'2026-07-01']
    baselines(feat,outcomes,test_sessions,model_root)
    stats,daily = summarize(trades,test_sessions,bootstrap=True)
    trades.to_parquet(model_root/'trade_ledger.parquet',index=False)
    stresses = {}
    for key,slip,latency in [('slippage_0.5pt',.5,250),('slippage_1pt',1.,250),('latency_1000ms',.25,1000)]:
        stress_trades,missing = [],[]
        for state in states:
            st,mi = stress_fold(state,dev_files,slip,latency)
            stress_trades.append(st)
            missing.extend(mi)
        stress = pd.concat(stress_trades,ignore_index=True)
        if len(stress):
            stress = stress.sort_values('ts')
        stress.to_parquet(model_root/f'stress_{key}.parquet',index=False)
        metric,_ = summarize(stress,test_sessions)
        metric['unresolved'] = len(missing)
        stresses[key] = metric
        progress(f'Execution stress {key}',.85,timer)
    conditions = dict(min_trades=stats['n_trades']>=100,profit_factor=(stats['profit_factor'] or 0)>=1.15,
                      positive_folds=sum(f['metrics']['net_usd']>0 for f in fold_records)>=2,
                      confidence=stats['daily_mean_95ci'][0]>0,
                      execution_stress=all(s['net_usd']>0 and s['unresolved']==0 for s in stresses.values()),
                      resolved=not any(f.get('unresolved') for f in fold_records),
                      no_material_quote_gap=bool(trades.empty or trades.max_quote_gap_seconds.max()<=10.))
    passed = all(conditions.values())
    gate = dict(verdict='DEVELOPMENT ACCEPTED' if passed else 'REJECTED — edge not established',
                conditions=conditions,holdout_status='not consulted; development gate failed' if not passed else 'eligible for one frozen evaluation')
    report(model_root/'report',stats,daily,fold_records,stresses,gate,timer)
    # Final reservation is deliberately fail-closed. The separate final step is implemented
    # only if development passes; never reinterpret a failed experiment as authorization to peek.
    if passed:
        dump(root/'holdout_ready.json',dict(files=[str(p) for p in held_files],status='FROZEN_PROTOCOL_READY'))
    refs = [DatasetRef(logical_id=f'NQ/tape/{r["session"]}',semantic_type='raw_source',
                       content_sha256=r['sha256'],location=r['path'],n_rows=r['n_prints']) for r in inventory]
    manifest = RunManifest(run_id=f'{OBJECTIVE}/{root.name}',objective_id=OBJECTIVE,
        hypothesis='Causal auction position and flow predict selectively tradable intraday NQ net return.',
        responsibility_versions={k:hashes[p] for k,p in [('features','src/features/nq_selective.py'),
            ('outcomes','src/backtest/nq_selective.py'),('model_strategy','scripts/research/train_nq_selective.py'),
            ('economic_evaluation','src/evaluation/economic.py')]},
        code=CodeIdentity(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=True),
        datasets=refs,config_snapshot=config,config_snapshot_resolved=True,
        validation=ValidationDesign(split='nested_expanding_calendar',n_folds=3,purge_sessions=1,
            weighting='equal observed sessions for inference',hold_out_dates=tuple(config['holdout']),
            fold_windows=tuple(fold_records)),seeds={'numpy':SEED,'xgboost':SEED},
        execution_assumptions=dict(asdict(ASSUMPTIONS),
                                   simulated_slippage_points_per_side=.25,simulated_latency_ms=250),
        decision_policy={k:config[k] for k in ['max_entries','max_losing_trades','daily_loss_r','thresholds']},
        calibration={'method':'inner temporal policy selection; expected returns are not calibrated probabilities'},
        artifacts={'report':['pass_1/model_selective_ev/report/summary.md'],
                   'trade_ledger':['pass_1/model_selective_ev/trade_ledger.parquet']},
        extra={'status':'EXPERIMENTAL','gate':gate,'source_hashes':hashes})
    manifest.write(root)
    print(json.dumps(dict(run=str(root),metrics=stats,gate=gate),indent=2),flush=True)
    progress('✅ Done',1.,timer)


if __name__ == '__main__':
    main()
