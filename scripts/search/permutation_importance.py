"""Group-level permutation importance for the v009 3-class model.

Replicates the trainer preprocessing pipeline exactly, then for each feature
group zeroes out those columns in X_test and measures the drop in:
  - cross-entropy loss
  - BIG_UP precision@tau=0.80
  - BIG_DOWN precision@tau=0.80

Usage:
    python scripts/permutation_importance.py [--checkpoint CKPT_DIR] [--start DATE] [--end DATE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/opt/vtech")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="data/legacy/checkpoints/model_v009_3class")
parser.add_argument("--start", default="2025-10-01")
parser.add_argument("--end", default="2025-12-31")
args = parser.parse_args()

os.chdir("/opt/vtech")

# ── imports ───────────────────────────────────────────────────────────────────
import tensorflow as tf  # noqa: E402
from src.models.attention_lstm import AttentionLayer  # noqa: E402 — must be imported before load_model
from src.training.config import MLConfig  # noqa: E402
from src.training.data_pipeline import DataPipeline  # noqa: E402
from src.training.labels import generate_labels  # noqa: E402
from src.training.trainer import FEATURE_GROUP_PREFIXES  # noqa: E402

tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── config ────────────────────────────────────────────────────────────────────
cfg = MLConfig.from_env()

CKPT = args.checkpoint
SEQ_LEN = cfg.model.sequence_length
Z_THRESH = 5.0
BIG_UP_IDX = 2
TAU = 0.80

# ── 1. Load features ──────────────────────────────────────────────────────────
print(f"\n⏳ Loading features {args.start} → {args.end}...")
t0 = time.time()
pipeline = DataPipeline("data", cfg.data)
features_df = pipeline.load_features(args.start, args.end)

# Replicate trainer's feature_cols logic (respect enabled_groups)
feature_cols = [c for c in features_df.columns if not c.startswith("label") and not c.startswith("_")]
enabled = cfg.features.enabled_groups
disabled_prefixes = [
    FEATURE_GROUP_PREFIXES[grp]
    for grp, on in enabled.items()
    if not on and grp in FEATURE_GROUP_PREFIXES
]
if disabled_prefixes:
    feature_cols = [c for c in feature_cols if not any(c.startswith(p) for p in disabled_prefixes)]

# ── 2. Labels ─────────────────────────────────────────────────────────────────
prices = features_df["_close"]
labels_df = generate_labels(prices, cfg.labels)
full_df = pd.concat([features_df[feature_cols], labels_df], axis=1).dropna(subset=["label"])

# ── 3. Split ──────────────────────────────────────────────────────────────────
train_df, val_df, test_df = [s.copy() for s in pipeline.split_temporal(full_df)]

# ── 4. Preprocess (identical to trainer) ─────────────────────────────────────
train_mean = train_df[feature_cols].mean()
train_std = train_df[feature_cols].std().replace(0, 1)
for df_part in [train_df, val_df, test_df]:
    clipped = df_part[feature_cols].clip(
        lower=(train_mean - Z_THRESH * train_std),
        upper=(train_mean + Z_THRESH * train_std),
        axis=1,
    )
    df_part[feature_cols] = clipped

pipeline.fit_scaler(train_df, feature_cols, scaler_type=cfg.data.scaler_type)
test_s = pipeline.transform(test_df)
test_s[feature_cols] = test_s[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# ── 5. Sequences ──────────────────────────────────────────────────────────────
X_test, y_test = DataPipeline.create_sequences(
    test_s[feature_cols].values, test_s["label"].values, SEQ_LEN
)
print(f"   X_test shape: {X_test.shape}  y_test shape: {y_test.shape}")
print(f"   Labels: BIG_DOWN={int((y_test==0).sum())}  NO_SIGNAL={int((y_test==1).sum())}  BIG_UP={int((y_test==2).sum())}")
print(f"   Preprocessing done in {time.time()-t0:.1f}s")

# ── 6. Load model ─────────────────────────────────────────────────────────────
print(f"\n⏳ Loading model from {CKPT}...")
from src.models.keras_layers import load_keras_model
model = load_keras_model(
    f"{CKPT}/model.keras",
    custom_objects={"AttentionLayer": AttentionLayer},
)

# ── 7. Baseline ───────────────────────────────────────────────────────────────
def precision_at_tau(probs: np.ndarray, y_true: np.ndarray, cls_idx: int, tau: float) -> tuple[float, int]:
    mask = probs[:, cls_idx] >= tau
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0
    prec = float((y_true[mask] == cls_idx).mean())
    return prec, n

def evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    probs = model.predict(X, verbose=0, batch_size=1024)  # shape (N, 3)
    # cross-entropy loss
    y_oh = np.eye(3)[y.astype(int)]
    ce = float(-np.mean(np.sum(y_oh * np.log(probs + 1e-9), axis=1)))
    prec_up, n_up = precision_at_tau(probs, y, BIG_UP_IDX, TAU)
    prec_dn, n_dn = precision_at_tau(probs, y, 0, TAU)
    return {"ce": ce, "prec_up": prec_up, "n_up": n_up, "prec_dn": prec_dn, "n_dn": n_dn}

print("\n⏳ Baseline evaluation...")
baseline = evaluate(X_test, y_test)
print(f"   Baseline CE={baseline['ce']:.4f}  BIG_UP precision@{TAU}={baseline['prec_up']:.4f} (n={baseline['n_up']})"
      f"  BIG_DOWN precision@{TAU}={baseline['prec_dn']:.4f} (n={baseline['n_dn']})")

# ── 8. Group-level permutation importance ────────────────────────────────────
# Map each feature column to its index in the feature_cols list
feat_to_idx = {f: i for i, f in enumerate(feature_cols)}

# Build group → list of column indices in the feature dimension of X_test
# (X_test shape: N, SEQ_LEN, n_features)
groups_present: dict[str, list[int]] = {}
for group_name, prefix in FEATURE_GROUP_PREFIXES.items():
    idxs = [feat_to_idx[f] for f in feature_cols if f.startswith(prefix) and f in feat_to_idx]
    if idxs:
        groups_present[group_name] = idxs

print(f"\n⏳ Running permutation importance over {len(groups_present)} groups...\n")
print(f"{'Group':<25} {'N_feats':>7}  {'ΔCE':>8}  {'Δprec_UP':>10}  {'Δprec_DN':>10}  {'n_UP':>6}  {'n_DN':>6}")
print("-" * 85)

results = []
for i, (group_name, col_idxs) in enumerate(sorted(groups_present.items())):
    # Zero out this group's features across all time steps
    X_perm = X_test.copy()
    X_perm[:, :, col_idxs] = 0.0

    m = evaluate(X_perm, y_test)
    delta_ce = m["ce"] - baseline["ce"]
    delta_up = baseline["prec_up"] - m["prec_up"]
    delta_dn = baseline["prec_dn"] - m["prec_dn"]

    results.append({
        "group": group_name,
        "n_feats": len(col_idxs),
        "delta_ce": delta_ce,
        "delta_prec_up": delta_up,
        "delta_prec_dn": delta_dn,
        "prec_up_zeroed": m["prec_up"],
        "n_up_zeroed": m["n_up"],
        "prec_dn_zeroed": m["prec_dn"],
        "n_dn_zeroed": m["n_dn"],
    })
    print(f"  [{i+1:2d}/{len(groups_present)}] {group_name:<22} {len(col_idxs):>7}  "
          f"{delta_ce:>+8.4f}  {delta_up:>+10.4f}  {delta_dn:>+10.4f}  "
          f"{m['n_up']:>6}  {m['n_dn']:>6}")

# ── 9. Ranked summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("RANKED BY ΔCE (higher = group hurt more when zeroed = more important)\n")
ranked = sorted(results, key=lambda r: -r["delta_ce"])
print(f"{'Rank':<5} {'Group':<25} {'N_feats':>7}  {'ΔCE':>8}  {'Δprec_UP':>10}  {'Δprec_DN':>10}")
print("-" * 70)
for rank, r in enumerate(ranked, 1):
    print(f"  {rank:<4} {r['group']:<25} {r['n_feats']:>7}  {r['delta_ce']:>+8.4f}  "
          f"{r['delta_up_label'] if 'delta_up_label' in r else r['delta_prec_up']:>+10.4f}  "
          f"{r['delta_prec_dn']:>+10.4f}")

print("\n")
print("RANKED BY Δprec_UP@0.80 (BIG_UP precision drop — most important for signal quality)\n")
ranked_up = sorted(results, key=lambda r: -r["delta_prec_up"])
print(f"{'Rank':<5} {'Group':<25} {'prec_UP_zeroed':>15}  {'Δprec_UP':>10}  {'n_UP_zeroed':>12}")
print("-" * 70)
for rank, r in enumerate(ranked_up, 1):
    print(f"  {rank:<4} {r['group']:<25} {r['prec_up_zeroed']:>15.4f}  "
          f"{r['delta_prec_up']:>+10.4f}  {r['n_up_zeroed']:>12}")

# ── 10. Save results ──────────────────────────────────────────────────────────
out_path = f"{CKPT}/permutation_importance.json"
out = {
    "baseline": baseline,
    "tau": TAU,
    "groups": results,
    "ranked_by_delta_ce": [r["group"] for r in ranked],
    "ranked_by_delta_prec_up": [r["group"] for r in ranked_up],
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n✅ Results saved to {out_path}")
print(f"   Total time: {time.time()-t0:.1f}s")
