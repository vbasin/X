"""Data pipeline: load Parquet features, split, preprocess, create sequences."""

from __future__ import annotations

from src.common.data_paths import (
    hive_timeframe_from_path,
    list_feature_parquet_files,
)

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.training.config import DataConfig

logger = logging.getLogger("vtech.training.data_pipeline")

SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a feature frame index to ``datetime64[ns, UTC]``.

    Historical parquets were written as ``Etc/UTC``; live rebuilds use ``UTC``.
    Those are equal offsets but distinct tz objects, so ``pd.concat`` produces
    an object index and ``DatetimeIndex(...)`` raises
    ``Tz-aware datetime.datetime cannot be converted to datetime64 unless utc=True``.
    """
    if df is None or df.empty:
        return df
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        if "bar_end_ts" in df.columns:
            df = df.set_index("bar_end_ts")
            idx = df.index
        elif "ts" in df.columns:
            df = df.set_index("ts")
            idx = df.index
        idx = pd.to_datetime(idx, utc=True)
    elif idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    if idx is df.index:
        return df
    out = df.copy(deep=False)
    out.index = idx
    return out


class DataPipeline:
    """Loads cached Parquet features and prepares them for training."""

    def __init__(self, data_dir: str | Path, config: DataConfig | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.config = config or DataConfig()
        self.scaler = None
        self.scaler_type: str = "robust"
        self.feature_columns: list[str] = []

    def _flat_feature_dir(self) -> Path:
        """Directory of dated ``features_YYYY-MM-DD.parquet`` files, if any.

        Used only for tmp/test day-file dirs. Production loads monthly parts.
        """
        d = self.data_dir
        if d.is_dir() and any(
            p.is_file() and len(p.stem) == len("features_YYYY-MM-DD")
            for p in d.glob("features_*.parquet")
        ):
            return d
        return d

    def _dated_day_files(self) -> bool:
        d = self.data_dir
        if not d.is_dir():
            return False
        return any(
            p.is_file() and len(p.stem) == len("features_YYYY-MM-DD")
            for p in d.glob("features_*.parquet")
        )

    def _try_hive_load(self, start_date: str, end_date: str):
        """Load monthly parts from *data_dir* or the 5s store.

        Only parts whose ``year=/month=`` partition intersects
        [*start_date*, *end_date*] are opened — never the full hive.
        Tmp/test dirs that hold dated day files stay on the flat path.
        """
        from src.common.data_paths import clip_hive_part_files, iter_hive_part_files
        from src.features.native_tf import concat_hive_parquets, load_hive_frame

        d = self.data_dir
        tf = hive_timeframe_from_path(d)
        if tf:
            files = list_feature_parquet_files(d)
            if files:
                clipped = clip_hive_part_files(files, start_date, end_date)
                if not clipped:
                    raise FileNotFoundError(
                        f"no {tf} hive parts in {start_date}→{end_date} under {d}"
                    )
                return concat_hive_parquets(clipped, start_date, end_date)
        if self._dated_day_files():
            return None
        try:
            if iter_hive_part_files("5s", start=start_date, end=end_date):
                return load_hive_frame("5s", start_date, end_date)
        except Exception:
            return None
        return None

    # ── load ──────────────────────────────────────────────────────

    def load_features(
        self,
        start_date: str,
        end_date: str,
        snapshot_path: "Path | str | None" = None,
    ) -> pd.DataFrame:
        """Load and concatenate daily Parquet feature files for [start, end).

        Parameters
        ----------
        start_date:
            Inclusive start date (ISO format ``YYYY-MM-DD``).
        end_date:
            Inclusive end date (ISO format ``YYYY-MM-DD``).
        snapshot_path:
            Optional path to a snapshot directory that follows the canonical
            structure produced by ``scripts/export_snapshot.py``::

                <snapshot_path>/date=YYYY-MM-DD/features.parquet

            When *None* (default), uses the hive (``timeframe={tf}/``) when
            *data_dir* is a hive tree or the canonical 5s cache, else dated
            day files under *data_dir*.
        """
        if snapshot_path is not None:
            snap_root = Path(snapshot_path)

            def _read_date(d: pd.Timestamp) -> pd.DataFrame | None:
                path = snap_root / f"date={d.date()}" / "features.parquet"
                if path.exists():
                    return ensure_utc_index(pd.read_parquet(path))
                logger.debug("missing_parquet", extra={"date": str(d.date())})
                return None

            src_desc = str(snap_root)
        else:
            hive_df = self._try_hive_load(start_date, end_date)
            if hive_df is not None:
                logger.info(
                    "loaded_features",
                    extra={"rows": len(hive_df), "cols": hive_df.shape[1], "src": "hive"},
                )
                return ensure_utc_index(hive_df)
            parquet_dir = self._flat_feature_dir()

            def _read_date(d: pd.Timestamp) -> pd.DataFrame | None:
                path = parquet_dir / f"features_{d.date()}.parquet"
                if path.exists():
                    return ensure_utc_index(pd.read_parquet(path))
                logger.debug("missing_parquet", extra={"date": str(d.date())})
                return None

            src_desc = str(parquet_dir)

        dates = pd.date_range(start_date, end_date, freq="B")  # business days

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(_read_date, dates))
        frames = [r for r in results if r is not None]

        if not frames:
            raise FileNotFoundError(f"No feature files found in {src_desc}")

        df = pd.concat(frames).sort_index()
        # Live runtime appends 08:00–08:29 CT onto yesterday's file until
        # trading_start (08:30); today's rebuild also has those bars. Concat
        # then has duplicate timestamps and sequence Backtest raises
        # "Must have equal len keys and value when setting with an iterable".
        # Keep the later file (canonical session day).
        dup = df.index.duplicated(keep="last")
        n_dup = int(dup.sum())
        if n_dup:
            logger.warning(
                "dropped_duplicate_feature_timestamps",
                extra={"n": n_dup, "start": start_date, "end": end_date},
            )
            df = df.loc[~dup]
        logger.info("loaded_features", extra={"rows": len(df), "cols": df.shape[1]})
        return df

    # ── split ─────────────────────────────────────────────────────

    def split_temporal(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal train/val/test split (no shuffle — preserves time order)."""
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = train_end + int(n * self.config.val_ratio)
        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    # ── preprocess ────────────────────────────────────────────────

    def fit_scaler(self, df: pd.DataFrame, feature_cols: list[str], scaler_type: str = "robust") -> None:
        """Fit a scaler on the training set."""
        self.feature_columns = feature_cols
        self.scaler_type = scaler_type
        scaler_cls = SCALERS.get(scaler_type, RobustScaler)
        self.scaler = scaler_cls()
        self.scaler.fit(df[feature_cols].values)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted scaler to feature columns."""
        if self.scaler is None:
            raise RuntimeError("Call fit_scaler() first")
        out = df.copy()
        out[self.feature_columns] = self.scaler.transform(df[self.feature_columns].values)
        return out

    # ── sequences ─────────────────────────────────────────────────

    @staticmethod
    def create_sequences(
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 128,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a window of *seq_len* over X to produce 3-D sequence arrays."""
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len : i])
            ys.append(y[i])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)
