"""On-the-fly 1s windows from :class:`~src.search.path1s_folds.PathSession`.

Materialising (N, seq_len, F) for every labelled entry would be terabytes.
This Keras Sequence copies one batch of causal windows from the session path.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.search.path1s_folds import PathSession


def entry_pairs(sessions: Sequence[PathSession], seq_len: int) -> list[tuple[int, int]]:
    """``(session_i, entry_j)`` where the window ``[t-seq_len+1, t]`` fits."""
    seq_len = int(seq_len)
    pairs: list[tuple[int, int]] = []
    for si, sess in enumerate(sessions):
        for ei, t in enumerate(sess.entry_idx):
            if int(t) + 1 >= seq_len:
                pairs.append((si, ei))
    return pairs


def gather_windows(
    sessions: Sequence[PathSession],
    pairs: Sequence[tuple[int, int]],
    seq_len: int,
) -> np.ndarray:
    seq_len = int(seq_len)
    f = int(sessions[0].X.shape[-1]) if sessions else 0
    out = np.empty((len(pairs), seq_len, f), dtype="float32")
    for i, (si, ei) in enumerate(pairs):
        t = int(sessions[si].entry_idx[ei])
        out[i] = sessions[si].X[t - seq_len + 1: t + 1]
    return out


def gather_field(
    sessions: Sequence[PathSession],
    pairs: Sequence[tuple[int, int]],
    field: str,
) -> np.ndarray:
    parts = [getattr(sessions[si], field)[ei] for si, ei in pairs]
    return np.stack(parts, axis=0) if parts else np.array([])


class PathWindowSequence:
    """Keras Sequence over session-local causal windows."""

    def __init__(
        self,
        sessions: Sequence[PathSession],
        seq_len: int,
        batch_size: int,
        *,
        y_fields: tuple[str, ...] = ("cr_up", "cr_down", "y_quantile"),
        sample_weight: np.ndarray | None = None,
        weight_fields: dict[str, str] | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.sessions = list(sessions)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.y_fields = y_fields
        self.sample_weight = sample_weight
        self.weight_fields = weight_fields or {
            "y_up": "cr_mask_up", "y_down": "cr_mask_down",
        }
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self.pairs = np.array(entry_pairs(self.sessions, self.seq_len), dtype=np.int32)
        if shuffle:
            self._rng.shuffle(self.pairs)

    def __len__(self) -> int:
        n = len(self.pairs)
        return max(1, (n + self.batch_size - 1) // self.batch_size) if n else 0

    def __getitem__(self, i: int):
        s = slice(i * self.batch_size, min((i + 1) * self.batch_size, len(self.pairs)))
        batch_pairs = [(int(a), int(b)) for a, b in self.pairs[s]]
        x = gather_windows(self.sessions, batch_pairs, self.seq_len)
        y = {name: gather_field(self.sessions, batch_pairs, name)
             for name in self.y_fields}
        y_out = {}
        if "cr_up" in y:
            y_out["y_up"] = y["cr_up"]
        elif "y_up" in y:
            y_out["y_up"] = y["y_up"]
        if "cr_down" in y:
            y_out["y_down"] = y["cr_down"]
        elif "y_down" in y:
            y_out["y_down"] = y["y_down"]
        if "y_quantile" in y:
            y_out["y_quantile"] = y["y_quantile"]
        weights: dict[str, np.ndarray] = {}
        for out_name, field in self.weight_fields.items():
            if out_name in y_out and self.sessions and getattr(self.sessions[0], field, None) is not None:
                weights[out_name] = gather_field(self.sessions, batch_pairs, field)
        if self.sample_weight is not None:
            sw = self.sample_weight[s]
            for k, w in list(weights.items()):
                if w.ndim == 2:
                    weights[k] = w * sw[:, None]
                else:
                    weights[k] = w * sw
            if "y_quantile" in y_out and "y_quantile" not in weights:
                weights["y_quantile"] = sw
            for k in y_out:
                if k not in weights:
                    weights[k] = sw
        if not weights:
            return x, y_out
        return x, y_out, weights

    def on_epoch_end(self) -> None:
        if self.shuffle and len(self.pairs):
            self._rng.shuffle(self.pairs)


# tf.keras.utils.Sequence subclass is created at init so tests can import this
# module without TensorFlow. Plugin code uses PathWindowSequence as a Sequence.
def as_keras_sequence(seq: PathWindowSequence) -> Any:
    import tensorflow as tf

    class _Wrapped(tf.keras.utils.Sequence):
        def __len__(self_inner):
            return seq.__len__()

        def __getitem__(self_inner, i):
            return seq.__getitem__(i)

        def on_epoch_end(self_inner):
            seq.on_epoch_end()

    return _Wrapped()
