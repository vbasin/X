#!/usr/bin/env python3
"""Fixed cash/long benchmark; never chooses a strategy or opens reserve files."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.research.run_nq_cadence_area import evaluate,dump,FEE


def baseline(data:pd.DataFrame,cost:float=1.)->pd.DataFrame:
    """One NQ long 09:00 next-minute open to 15:00 final close, causal fixed times."""
    entry=data[data.minute.eq(541)].set_index('session')[['open']].rename(columns={'open':'entry_raw'})
    end=data[data.minute.eq(900)].set_index('session')[['close']].rename(columns={'close':'exit_raw'})
    days=pd.DataFrame(index=pd.Index(sorted(data.session.unique()),name='session'))
    tr=days.join(entry).join(end);tr['valid']=tr.entry_raw.notna()&tr.exit_raw.notna()
    tr['entry_price']=tr.entry_raw+cost/2;tr['exit_price']=tr.exit_raw-cost/2
    tr['net_usd']=(tr.exit_price-tr.entry_price)*20-FEE
    tr.loc[~tr.valid,'net_usd']=np.nan
    return tr.reset_index()


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    if not (args.run/'frozen_policy.json').exists():raise ValueError('freeze first')
    dest=args.run/'benchmarks';dest.mkdir(exist_ok=False)
    data=pd.read_parquet(args.run/'training_matrix.parquet');data=data[data.session>='2021-01-01']
    summaries={}
    for cost in [1.,2.,4.]:
        tr=baseline(data,cost);stats,daily=evaluate(tr,sorted(data.session.unique()),bootstrap=True)
        tr.to_csv(dest/f'long_{cost:g}_trades.csv',index=False);daily.to_csv(dest/f'long_{cost:g}_daily.csv');summaries[str(cost)]=stats
    dump(dest/'summary.json',{'long_09_15':summaries,'cash':{'net_usd':0,'max_drawdown_usd':0}})
    print(summaries['1.0'])


if __name__=='__main__':main()
