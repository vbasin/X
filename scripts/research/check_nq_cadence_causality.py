#!/usr/bin/env python3
"""Real-data prefix and adversarial-future checks on a frozen policy artifact."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from src.common.vm_config import get_vm_config
from src.features.nq_noise_area import attach_noise_area
from src.backtest.nq_cadence_area import simulate_session
from scripts.research.run_nq_cadence_area import dump,FEE


def check(task:tuple)->dict:
    raw,session,cutoff,policy=task
    full=attach_noise_area(raw)
    prefix=attach_noise_area(raw.loc[raw.index<=cutoff])
    future=raw.copy();mask=future.index>cutoff
    future.loc[mask,['open','high','low','close']]*=1.1
    future.loc[mask,'volume']*=7
    perturbed=attach_noise_area(future)
    cols=['cash_open','sigma','upper','lower']
    for compared in [full,perturbed]:
        pd.testing.assert_frame_equal(prefix[cols],compared.loc[prefix.index,cols])
    day=full[full.session.eq(session)];short=prefix[prefix.session.eq(session)];changed=perturbed[perturbed.session.eq(session)]
    def book(frame:pd.DataFrame)->pd.DataFrame:
        m=policy['multiplier'];kw={k:v for k,v in policy.items() if k!='multiplier'}
        upper=frame.cash_open.to_numpy()*(1+m*frame.sigma.to_numpy())
        lower=frame.cash_open.to_numpy()*(1-m*frame.sigma.to_numpy())
        trades=simulate_session(frame,upper,lower,roundtrip_cost_points=1.,commission=FEE,**kw)
        for column in ['entry_decision_ts','entry_ts','exit_ts']:
            trades[column]=pd.to_datetime(trades[column],utc=True)
        return trades
    books=[book(v) for v in [day,short,changed]]
    original=books[0]
    for altered in books[1:]:
        a=original[original.entry_ts<cutoff][['entry_decision_ts','entry_ts','side','entry_price','stop_initial']].reset_index(drop=True)
        b=altered[altered.entry_ts<cutoff][a.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_dtype=False)
        a=original[original.valid & (original.exit_ts<cutoff)].reset_index(drop=True)
        b=altered[altered.valid & (altered.exit_ts<cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_dtype=False)
    return dict(session=session,cutoff=str(cutoff),feature_prefix=True,adversarial_future=True,entry_prefix=True,completed_exit_prefix=True)


def main()->None:
    import json
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    policy=json.loads((args.run/'frozen_policy.json').read_text())['selected']['policy']
    data=pd.read_parquet(args.run/'training_matrix.parquet');sessions=sorted(data.session.unique())
    tasks=[]
    for date in ['2012-06-05','2016-09-21','2019-08-05','2020-03-24']:
        if date not in sessions:continue
        i=sessions.index(date);subset=data[data.session.isin(sessions[i-14:i+1])][['open','high','low','close','volume']].copy()
        for clock in ['10:00','13:00']:
            tasks.append((subset,date,pd.Timestamp(date+' '+clock,tz='America/Chicago').tz_convert('UTC'),policy))
    with ProcessPoolExecutor(max_workers=get_vm_config().workers_for(mem_per_worker_gb=2.)) as pool:results=list(pool.map(check,tasks))
    if len(results)!=8:raise AssertionError('expected eight real-data checks')
    dump(args.run/'causality_checks.json',dict(passed=True,checks=results,scope='observed discovery dates; features include cutoff; fill comparisons strictly before cutoff because next-bar opens at cutoff belong to a not-yet-closed bar; not proof against every possible data issue'))
    print('8 real-data prefix and adversarial-future checks passed')


if __name__=='__main__':main()
