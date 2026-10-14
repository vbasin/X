#!/usr/bin/env python3
"""Frozen-policy recent quote compatibility, then one gated final reserve replay."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from src.common.vm_config import get_vm_config
from src.features.tape import list_tape_files
from src.features.nq_noise_area import attach_noise_area,eligible_sessions
from src.backtest.nq_cadence_quotes import simulate_quote_session
from src.backtest.nq_cadence_area import simulate_session
from src.evaluation.economic import (Trade,Leg,LegRole,OrderType,compute_trade_ledger,load_assumptions)
from scripts.research.run_nq_cadence_area import evaluate,dump


def read_tape_bars(pathstr:str)->tuple[pd.DataFrame,dict]:
    p=Path(pathstr);tape=pd.read_parquet(p)
    if not tape.index.is_monotonic_increasing:raise ValueError('tape ordering')
    bars=tape.resample('1min',closed='left',label='right').agg(open=('price','first'),high=('price','max'),low=('price','min'),close=('price','last'),volume=('size','sum'))
    bars=bars.dropna(subset=['close']);bars=bars.loc[bars.index<=tape.index[-1]];bars.index.name='ts';bars['session']=p.stem[5:]
    return bars,dict(path=str(p),session=p.stem[5:],sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(tape))


def quote_day(task:tuple)->pd.DataFrame:
    path,bars,policy,slip,latency,fee=task
    tape=pd.read_parquet(path)
    m=policy['multiplier'];kw={k:v for k,v in policy.items() if k!='multiplier'}
    upper=bars.cash_open.to_numpy()*(1+m*bars.sigma.to_numpy())
    lower=bars.cash_open.to_numpy()*(1-m*bars.sigma.to_numpy())
    tr=simulate_quote_session(tape,bars,upper,lower,slippage_points=slip,latency_ms=latency,commission=fee,**kw)
    tr['session']=Path(path).stem[5:]
    return tr


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--phase',choices=['compatibility','holdout'],default='compatibility');ap.add_argument('--context',type=Path,required=True);args=ap.parse_args()
    timer=time.monotonic();root=args.run;phase=args.phase;dest=root/phase
    if dest.exists():raise FileExistsError('preserving prior evaluation; do not repeat final reserve')
    frozen=json.loads((root/'frozen_policy.json').read_text());policy=frozen['selected']['policy']
    historical_gate=json.loads((root/'validation_gate.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    historical_sources=json.loads((root/'source_hashes.json').read_text())
    source_paths=['scripts/research/eval_nq_cadence_quotes.py',
                  'src/backtest/nq_cadence_quotes.py','src/features/nq_noise_area.py',
                  'scripts/research/run_nq_cadence_area.py','src/backtest/nq_cadence_area.py',
                  'src/evaluation/economic.py','config/execution_costs.yaml','src/features/tape.py']
    code_bytes={source:Path(source).read_bytes() for source in source_paths}
    code_hashes={source:hashlib.sha256(value).hexdigest() for source,value in code_bytes.items()}
    feature_source='src/features/nq_noise_area.py'
    if code_hashes[feature_source]!=historical_sources[feature_source]:
        raise ValueError('feature implementation differs from frozen historical source')
    economics=load_assumptions('config/execution_costs.yaml','NQ')
    fee=economics.commission_per_trade
    if fee!=float(protocol['commission']):
        raise ValueError('commission differs from frozen historical protocol')
    scenarios=[('base',.25,250),('slip_0.5',.5,250),('slip_1',1.,250),('latency_1000ms',.25,1000)]
    if phase=='holdout':
        compatible=json.loads((root/'compatibility/gate.json').read_text())
        if not historical_gate['passed'] or not compatible['passed']:raise ValueError('development/compatibility gate failed; reserve locked')
    dest.mkdir(parents=True)
    # Write irreversible consultation marker before any reserved raw price read.
    spec=dict(phase=phase,policy=policy,historical_gate=historical_gate,
              frozen_policy_sha256=hashlib.sha256((root/'frozen_policy.json').read_bytes()).hexdigest(),
              reserved_read=phase=='holdout',created_at=str(pd.Timestamp.now(tz='UTC')),
              code_sha256=code_hashes,
              context_path=str(args.context.resolve()),
              execution={'instrument':'NQ','point_value':economics.contract_multiplier,
                         'commission_per_trade':fee,'spread':'observed pretrade BBO proxy',
                         'fill_model':'first subsequent valid print quote after latency; resolved fills',
                         'scenarios':[dict(name=n,slippage_points_per_side=s,latency_ms=l) for n,s,l in scenarios]},
              frozen_feature_hash_verified=True,frozen_commission_verified=True)
    dump(dest/'evaluation_lock.json',spec)
    for source,value in code_bytes.items():
        p=dest/'code'/source;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
    context=pd.read_parquet(args.context)
    context_record=dict(path=str(args.context.resolve()),
                        sha256=hashlib.sha256(args.context.read_bytes()).hexdigest(),
                        rows=len(context),first_session=str(context.session.min()),
                        last_session=str(context.session.max()))
    upstream=[]
    for filename in ['build_manifest.json','source_inventory.parquet']:
        source=args.context.parent/filename
        if source.exists():
            value=source.read_bytes();target=dest/'context_provenance'/filename
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
            upstream.append(dict(path=str(source.resolve()),sha256=hashlib.sha256(value).hexdigest()))
    context_record['upstream_sidecars']=upstream
    files=[p for p in list_tape_files() if p.stem[5:]<=('2026-09-22' if phase=='holdout' else '2026-06-30')]
    allowed=eligible_sessions('2025-01-01','2026-09-22');files=[p for p in files if p.stem[5:] in allowed]
    workers=get_vm_config().workers_for(mem_per_worker_gb=2.)
    frames=[];inventory=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(read_tape_bars,str(p)) for p in files]
        for i,job in enumerate(as_completed(jobs),1):
            f,inv=job.result();frames.append(f);inventory.append(inv)
            if i%50==0 or i==len(jobs):print(f'[{"█"*int(20*i/len(jobs))}{"░"*(20-int(20*i/len(jobs)))}] quote features {i}/{len(jobs)}',flush=True)
    context=context[context.session.isin(allowed)]
    first=min(p.stem[5:] for p in files);context=context[context.session<first]
    context_record['used_rows']=len(context)
    context_record['used_first_session']=str(context.session.min())
    context_record['used_last_session']=str(context.session.max())
    context_record['strictly_before_first_tape_session']=first
    dump(dest/'context_provenance.json',context_record)
    raw=pd.concat([context,*frames]).sort_index();features=attach_noise_area(raw)
    features.to_parquet(dest/'training_matrix.parquet');dump(dest/'input_inventory.json',sorted(inventory,key=lambda r:r['session']))
    sessions=sorted(p.stem[5:] for p in files if (p.stem[5:]<'2026-07-01')==(phase=='compatibility'))
    groups={s:g for s,g in features.groupby('session') if s in sessions}
    test_files=[p for p in files if p.stem[5:] in sessions]
    if set(sessions)!=set(groups):raise AssertionError('missing feature session')
    summaries={}
    for name,slip,latency in scenarios:
        tasks=[(str(p),groups[p.stem[5:]],policy,slip,latency,fee) for p in test_files]
        parts=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for tr in pool.map(quote_day,tasks):parts.append(tr)
        trades=pd.concat(parts,ignore_index=True).sort_values('entry_ts')
        valid=trades[trades.valid]
        # Canonical accounting accepts resolved fills without double-slipping.
        ledger=compute_trade_ledger([Trade(str(i),int(r.side),
            Leg(OrderType.MARKET,resolved_fill_price=r.entry_price,fill_source='pretrade_BBO_proxy_plus_latency_slippage',ts=str(r.entry_ts)),
            Leg(OrderType.MARKET,role=LegRole.EXIT_TIME,resolved_fill_price=r.exit_price,fill_source='pretrade_BBO_proxy_plus_latency_slippage',ts=str(r.exit_ts)))
            for i,r in enumerate(valid.itertuples())],economics)
        np.testing.assert_allclose([r.net_dollars for r in ledger.trades],valid.net_usd,atol=1e-7)
        stats,daily=evaluate(trades,sessions,bootstrap=True)
        stats['yearly_net_usd']=daily.assign(year=daily.index.str[:4]).groupby('year').net_usd.sum().to_dict()
        p=dest/name;p.mkdir();trades.to_parquet(p/'trade_ledger.parquet',index=False);daily.to_csv(p/'daily.csv');dump(p/'metrics.json',stats)
        summaries[name]=stats
        print(name,json.dumps(stats),flush=True)
    b=summaries['base']
    gate=dict(positive=b['net_usd']>0,profit_factor=(b['profit_factor'] or 0)>=1.10,
              stress_positive=all(s['net_usd']>0 for s in summaries.values()),resolved=all(s['unresolved']==0 for s in summaries.values()))
    if phase=='holdout':gate['confidence']=b['daily_mean_95ci'][0]>0
    gate['passed']=all(gate.values());dump(dest/'gate.json',gate);dump(dest/'summary.json',summaries)
    print('GATE',gate,flush=True);print(f'✅ Done in {time.monotonic()-timer:.1f}s',flush=True)


if __name__=='__main__':main()
