"""Feature parquet load (duplicate timestamps from live next-session spill)."""

from __future__ import annotations

import pandas as pd

from src.training.data_pipeline import DataPipeline


def test_load_features_keeps_later_file_on_overlap(tmp_path):
    a = pd.DataFrame(
        {"cs_close": [1.0, 2.0]},
        index=pd.to_datetime(
            ["2026-08-25 19:44:55+00:00", "2026-08-26 13:02:00+00:00"]
        ),
    )
    b = pd.DataFrame(
        {"cs_close": [9.0, 3.0]},
        index=pd.to_datetime(
            ["2026-08-26 13:02:00+00:00", "2026-08-26 13:05:00+00:00"]
        ),
    )
    a.to_parquet(tmp_path / "features_2026-08-25.parquet")
    b.to_parquet(tmp_path / "features_2026-08-26.parquet")
    df = DataPipeline(tmp_path).load_features("2026-08-25", "2026-08-26")
    assert df.index.is_unique
    ts = pd.Timestamp("2026-08-26 13:02:00", tz="UTC")
    assert float(df.loc[ts, "cs_close"]) == 9.0
    assert len(df) == 3
