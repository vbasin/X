#!/usr/bin/env python3
"""Long-history economic discovery with one frozen validation configuration."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time
from typing import Any
import numpy as np
import pandas as pd
from src.common.run_layout import new_run_ts,run_dir
from src.common.vm_config import get_vm_config
from src.features.nq_noise_area import attach_noise_area,eligible_sessions
from src.backtest.nq_cadence_area import simulate_session
from src.evaluation.economic import load_assumptions
from src.experiment.manifest import RunManifest,DatasetRef,CodeIdentity,ValidationDesign
from src.common.data_paths import logical_path

OBJECTIVE='nq-cadence-area-continuation'
FEE=load_assumptions('config/execution_costs.yaml','NQ').commission_per_trade
POLICIES=[dict(multiplier=m,cadence_minutes=c,direction_mode=d,use_vwap=True,max_entries=3,emergency_stop=e)
          for m,c,d,e in itertools.product([.75,1.,1.25,1.5,2.,2.5,3.],[30,60],['both','long_only'],[False,True])]
SOURCE_FILES=['scripts/research/run_nq_cadence_area.py','src/features/nq_noise_area.py',
              'src/backtest/nq_cadence_area.py','docs/research/NQ_CADENCE_AREA_PROTOCOL.md',
              'config/execution_costs.yaml','scripts/research/build_nq_long_bars.py']
SEED=20260924


def dump(p:Path,x:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n')


def evaluate(trades:pd.DataFrame,sessions:list[str],bootstrap:bool=False)->tuple[dict,pd.DataFrame]:
    valid=trades[trades.valid] if len(trades) else trades
    daily=pd.DataFrame(index=pd.Index(sessions,name='session'))
    daily['net_usd']=valid.groupby('session').net_usd.sum() if len(valid) else 0.
    daily['net_usd']=daily.net_usd.fillna(0.)
    daily['equity_usd']=daily.net_usd.cumsum()
    daily['drawdown_usd']=daily.equity_usd-daily.equity_usd.cummax().clip(lower=0.)
    pnl=valid.net_usd.to_numpy() if len(valid) else np.empty(0)
    loss=pnl[pnl<0];wins=pnl[pnl>0]
    eq=np.r_[0.,pnl.cumsum()]
    streak=longest=0
    for v in pnl:
        streak=streak+1 if v<0 else 0;longest=max(longest,streak)
    std=daily.net_usd.std(ddof=1)
    result=dict(n_sessions=len(sessions),n_trades=len(pnl),unresolved=int(len(trades)-len(valid)),
                net_usd=float(pnl.sum()),profit_factor=float(wins.sum()/-loss.sum()) if len(loss) else None,
                win_rate=float((pnl>0).mean()) if len(pnl) else 0.,loss_count=len(loss),
                total_loss_usd=float(-loss.sum()),avg_trade_usd=float(pnl.mean()) if len(pnl) else 0.,
                max_drawdown_usd=float(np.max(np.maximum.accumulate(eq)-eq)),
                daily_drawdown_usd=float(-daily.drawdown_usd.min()),
                worst_trade_usd=float(pnl.min()) if len(pnl) else 0.,worst_day_usd=float(daily.net_usd.min()),
                longest_losing_streak=longest,sharpe_daily=float(daily.net_usd.mean()/std*np.sqrt(252)) if std>0 else 0.,
                mean_daily_usd=float(daily.net_usd.mean()))
    if bootstrap:
        rng=np.random.default_rng(SEED);v=daily.net_usd.to_numpy();n=len(v)
        idx=(rng.integers(0,n,(5000,(n+4)//5,1))+np.arange(5))%n
        means=v[idx.reshape(5000,-1)[:,:n]].mean(axis=1)
        result['daily_mean_95ci']=np.quantile(means,[.025,.975]).tolist()
    return result,daily


def run_day(task:tuple)->dict[int,pd.DataFrame]:
    session,bars,policies,cost=task
    result={}
    for i,p in policies:
        m=p['multiplier'];kw={k:v for k,v in p.items() if k!='multiplier'}
        upper=bars.cash_open.to_numpy()*(1+m*bars.sigma.to_numpy())
        lower=bars.cash_open.to_numpy()*(1-m*bars.sigma.to_numpy())
        tr=simulate_session(bars,upper,lower,roundtrip_cost_points=cost,commission=FEE,**kw)
        tr['session']=session;tr['policy_id']=i
        result[i]=tr
    return result


def run_book(data:pd.DataFrame,policies:list[tuple[int,dict]],cost:float,timer:float)->dict[int,pd.DataFrame]:
    results={i:[] for i,_ in policies}
    days=list(data.groupby('session',sort=True))
    with ProcessPoolExecutor(max_workers=get_vm_config().workers_for(mem_per_worker_gb=2.0)) as pool:
        jobs=[pool.submit(run_day,(s,g,policies,cost)) for s,g in days]
        for n,job in enumerate(as_completed(jobs),1):
            for i,tr in job.result().items():results[i].append(tr)
            if n%250==0 or n==len(days):
                k=int(20*n/len(days));print(f'[{"█"*k}{"░"*(20-k)}] {n}/{len(days)} sessions cost={cost} ({time.monotonic()-timer:.0f}s)',flush=True)
    return {i:pd.concat(v,ignore_index=True).sort_values('entry_ts') for i,v in results.items()}


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--bars',type=Path,required=True);args=ap.parse_args()
    timer=time.monotonic();root=run_dir(OBJECTIVE,new_run_ts());root.mkdir(parents=True)
    print(f'RUN_DIR={root}',flush=True)
    source_hashes={}
    for file in SOURCE_FILES:
        raw=Path(file).read_bytes();source_hashes[file]=hashlib.sha256(raw).hexdigest()
        p=root/'code'/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    dump(root/'source_hashes.json',source_hashes)
    config=dict(policies=POLICIES,lookback_sessions=14,discovery=['2011-01-01','2020-12-31'],
                validation=['2021-01-01','2024-12-31'],holdout=['2026-07-01','2026-09-22'],
                cost_points=[1.,2.,4.],commission=FEE,seed=SEED,
                calendar='XNYS full cash sessions, exclude quarterly third-Friday Mon-Fri week')
    dump(root/'protocol.json',config)
    import exchange_calendars
    dump(root/'environment.json',{'exchange_calendars':exchange_calendars.__version__,'pandas':pd.__version__,'numpy':np.__version__})
    raw=pd.read_parquet(args.bars)
    aggregate_rows=len(raw)
    for sidecar in ['build_manifest.json','source_inventory.parquet']:
        source=args.bars.parent/sidecar
        if source.exists():
            target=root/'upstream'/sidecar;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    allowed=eligible_sessions('2011-01-01','2024-12-31')
    present=sorted(set(raw.session)&set(allowed))
    raw=raw[raw.session.isin(present)]
    data=attach_noise_area(raw)
    data.to_parquet(root/'training_matrix.parquet')
    discovery=data[data.session<'2021-01-01'];disc_sessions=[s for s in present if s<'2021-01-01']
    books=run_book(discovery,list(enumerate(POLICIES)),1.,timer)
    stress_books=run_book(discovery,list(enumerate(POLICIES)),2.,timer)
    metrics=[]
    for i,policy in enumerate(POLICIES):
        tr=books[i];stats,daily=evaluate(tr,disc_sessions)
        yearly=daily.assign(year=daily.index.str[:4]).groupby('year').net_usd.sum()
        halves=[]
        for start,end in [('2011-01-01','2016-01-01'),('2016-01-01','2021-01-01')]:
            sub=tr[(tr.session>=start)&(tr.session<end)];ss=[s for s in disc_sessions if start<=s<end]
            halves.append(evaluate(sub,ss)[0])
        stress_half=[]
        for start,end in [('2011-01-01','2016-01-01'),('2016-01-01','2021-01-01')]:
            st=stress_books[i];st=st[(st.session>=start)&(st.session<end)]
            stress_half.append(evaluate(st,[d for d in disc_sessions if start<=d<end])[0]['net_usd'])
        qualifies=all(v>0 for v in stress_half) and (yearly>0).sum()>=6 and (stats['profit_factor'] or 0)>=1.10 and stats['n_trades']>=300 and stats['unresolved']==0 and int((~stress_books[i].valid).sum())==0
        metrics.append(dict(policy_id=i,policy=policy,metrics=stats,halves=halves,
                            discovery_stress_half_net=stress_half,positive_years=int((yearly>0).sum()),qualifies=bool(qualifies),
                            score=min(h['sharpe_daily'] for h in halves)))
        p=root/'discovery'/f'policy_{i}';p.mkdir(parents=True)
        tr.to_parquet(p/'trade_ledger.parquet',index=False);daily.to_csv(p/'daily.csv')
        stress_books[i].to_parquet(p/'stress_2pt.parquet',index=False)
    dump(root/'discovery_trials.json',metrics)
    qualified=[r for r in metrics if r['qualifies']]
    selected=max(qualified or metrics,key=lambda r:r['score'])
    frozen=dict(selected=selected,selection_passed=bool(qualified),frozen_before_validation=True)
    dump(root/'frozen_policy.json',frozen)
    print('FROZEN',json.dumps(frozen),flush=True)
    policy_id=selected['policy_id'];policy=POLICIES[policy_id]
    summaries={}
    # Historical 2021-24 is opened only for the one selected configuration.
    valid=data[data.session>='2021-01-01'];validation_sessions=[s for s in present if s>='2021-01-01']
    for cost in [1.,2.,4.]:
        book=run_book(valid,[(policy_id,policy)],cost,timer)[policy_id]
        summary,daily=evaluate(book,validation_sessions,bootstrap=True)
        yearly=daily.assign(year=daily.index.str[:4]).groupby('year').net_usd.sum().to_dict()
        summary['yearly_net_usd']=yearly
        p=root/'validation'/f'cost_{cost:g}';p.mkdir(parents=True)
        book.to_parquet(p/'trade_ledger.parquet',index=False);daily.to_csv(p/'daily.csv');dump(p/'metrics.json',summary)
        summaries[str(cost)]=summary
    base=summaries['1.0'];stress=summaries['2.0']
    gate=dict(discovery_qualified=bool(qualified),validation_positive_halves=(sum(v for k,v in base['yearly_net_usd'].items() if k<'2023')>0 and sum(v for k,v in base['yearly_net_usd'].items() if k>='2023')>0),
              validation_pf=(base['profit_factor'] or 0)>=1.10,validation_confidence=base['daily_mean_95ci'][0]>0,
              validation_cost_stress=stress['net_usd']>0,resolved=all(v['unresolved']==0 for v in summaries.values()))
    gate['passed']=all(gate.values());gate['holdout_read']=False
    dump(root/'validation_gate.json',gate)
    content=hashlib.sha256(args.bars.read_bytes()).hexdigest()
    manifest=RunManifest(run_id=f'{OBJECTIVE}/{root.name}',objective_id=OBJECTIVE,
        hypothesis='Large departures from prior-session same-clock movement continue sufficiently to pay costs with cadence-confirmed exits and optional fixed emergency stop.',
        responsibility_versions={'features':source_hashes['src/features/nq_noise_area.py'],'economic_evaluation':source_hashes['src/backtest/nq_cadence_area.py']},
        code=CodeIdentity(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=True),
        datasets=[DatasetRef(logical_id='NQ/calendar_front/1m/2011-2024',semantic_type='raw_source',content_sha256=content,location=logical_path(args.bars),n_rows=aggregate_rows)],
        config_snapshot=config,config_snapshot_resolved=True,validation=ValidationDesign(split='chronological_discovery_frozen_validation',n_folds=1,purge_sessions=0,hold_out_dates=tuple(config['holdout'])),
        seeds={'bootstrap':SEED},execution_assumptions={'historical_quote_data':False,'commission_per_trade':FEE,'roundtrip_proxy_points':1.,'point_value':20.},
        decision_policy=policy,calibration={'method':'no probabilistic model; prior-session empirical envelope'},extra={'gate':gate,'validation':summaries})
    manifest.write(root)
    print(json.dumps({'run':str(root),'selected':policy,'validation':summaries,'gate':gate},indent=2),flush=True)
    print(f'✅ Done in {time.monotonic()-timer:.1f}s',flush=True)


if __name__=='__main__':main()
