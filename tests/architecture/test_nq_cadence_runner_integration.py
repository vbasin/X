"""Runner policy propagation and immutable-hypothesis preflight checks."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.research import eval_nq_cadence_quotes as quote_runner
from scripts.research.run_nq_cadence_area import run_day


def ts(clock):
    return pd.Timestamp(f'2026-01-05 {clock}',tz='America/Chicago').tz_convert('UTC')


def fixture_bars():
    x=np.array([100.,112.,112.,115.])
    b=pd.DataFrame({'open':x,'high':x+1,'low':x-1,'close':x,'volume':1.,
                    'cash_open':100.,'sigma':.1},
                   index=pd.DatetimeIndex([ts(c) for c in ['08:31','09:00','09:01','15:00']]))
    b.loc[ts('09:01'),'low']=89.
    return b


def policy(emergency):
    return dict(multiplier=1.,cadence_minutes=30,direction_mode='long_only',
                use_vwap=True,max_entries=3,emergency_stop=emergency)


def test_policy_emergency_flag_reaches_both_execution_engines(tmp_path):
    b=fixture_bars()
    path=tmp_path/'tape_2026-01-05.parquet'
    bid=np.array([112.,89.,88.,115.])
    t=pd.DataFrame({'bid_px':bid,'ask_px':bid+.25},
        index=pd.DatetimeIndex([ts(c) for c in ['09:00:00.250','09:01:00','09:01:00.250','15:00:00.250']]))
    t.to_parquet(path)
    for emergency,reason in [(True,'emergency_stop'),(False,'session_close')]:
        p=policy(emergency)
        historical=run_day(('2026-01-05',b,[(0,p)],1.))[0]
        quotes=quote_runner.quote_day((str(path),b,p,.25,250,3.36))
        assert len(historical)==len(quotes)==1
        assert historical.iloc[0].exit_reason==quotes.iloc[0].exit_reason==reason
        if not emergency:
            assert historical.iloc[0].exit_ts==ts('15:00')
            assert quotes.iloc[0].exit_ts==ts('15:00:00.250')
            assert np.isnan(quotes.iloc[0].stop_initial)


@pytest.mark.parametrize('changed',['feature','fee'])
def test_quote_runner_refuses_changed_frozen_inputs_before_data_reads(tmp_path,monkeypatch,changed):
    root=tmp_path/'synthetic_run';root.mkdir()
    (root/'frozen_policy.json').write_text(json.dumps({'selected':{'policy':policy(True)}}))
    (root/'validation_gate.json').write_text(json.dumps({'passed':False}))
    (root/'protocol.json').write_text(json.dumps({'commission':4. if changed=='fee' else 3.36}))
    source='src/features/nq_noise_area.py'
    digest=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    (root/'source_hashes.json').write_text(json.dumps({source:'changed' if changed=='feature' else digest}))
    monkeypatch.setattr(sys,'argv',['eval',str(root),'--context',str(tmp_path/'missing_context.parquet')])
    def refuse_read(*args,**kwargs):
        raise AssertionError('no market data should be read before freeze checks pass')
    monkeypatch.setattr(quote_runner.pd,'read_parquet',refuse_read)
    with pytest.raises(ValueError,match='differs from frozen'):
        quote_runner.main()
    assert not (root/'compatibility').exists()
