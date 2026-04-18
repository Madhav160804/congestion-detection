#!/usr/bin/env python3
"""
K-Means Congestion Labeler
===========================
Replaces the rule-based HF-CEF labels in labeled_dataset.csv with
data-driven labels produced by K-Means clustering on host-observable
congestion signals.

Clustering features (all from iperf3 / ping — host side only):
    1. rtt_relative_iperf          — RTT inflation ratio (current / baseline)
    2. iperf_rttvar_us_mean        — RTT variance / jitter in microseconds
    3. iperf_achieved_ratio_capped — Achieved BW / target BW (capped at 1.0)

Cluster → Label mapping (by RTT ratio ordering of centroids):
    Lowest  RTT_ratio → Normal    (0)
    Middle  RTT_ratio → Onset     (1)  →  binary = 1
    Highest RTT_ratio → Congested (2)  →  binary = 1

This replaces the hardcoded threshold-based labeling with a fully
data-driven approach, ensuring every label boundary is learned from
the actual traffic distribution.

Usage:
    python3 dataset/kmeans_labeler.py
    # then retrain: python3 models/train_model.py
    # or demo:      python3 demo/train_demo_model.py
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DATASET_IN  = os.path.join(ROOT, "dataset", "labeled_dataset.csv")
DATASET_OUT = os.path.join(ROOT, "dataset", "labeled_dataset.csv")   # overwrite in-place
REPORT_OUT  = os.path.join(ROOT, "dataset", "kmeans_labeling_report.txt")

FEATURES = [
    "rtt_relative_iperf",
    "iperf_rttvar_us_mean",
    "iperf_achieved_ratio_capped",
]

K = 3
RANDOM_STATE = 42


def map_clusters_to_labels(kmeans, scaler, feature_matrix):
    """
    Determine which cluster ID corresponds to Normal / Onset / Congested
    by comparing the un-scaled centroids on RTT ratio (the primary signal).

    Returns: dict {cluster_id: ("Normal"|"Onset"|"Congested", binary_int)}
    """
    # Un-scale centroids back to original feature space
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    # Column order: RTT_ratio, jitter, BW_ratio
    rtt_col = 0

    # Sort cluster IDs by RTT ratio ascending: Normal < Onset < Congested
    order = np.argsort(centroids[:, rtt_col])   # ascending RTT

    label_names = {
        order[0]: ("Normal",    0),
        order[1]: ("Onset",     1),
        order[2]: ("Congested", 1),
    }
    return label_names, centroids, order


def main():
    print("=" * 60)
    print("  K-Means Congestion Labeler")
    print("=" * 60)

    # ── Load dataset ────────────────────────────────────────────
    print(f"\nLoading:  {DATASET_IN}")
    df = pd.read_csv(DATASET_IN)
    print(f"  Total rows : {len(df)}")

    # ── Show old label distribution ─────────────────────────────
    if "congestion_binary" in df.columns:
        old_counts = df["congestion_binary"].value_counts().sort_index()
        print(f"\n  Old label distribution (HF-CEF rule-based):")
        total = len(df)
        for val, cnt in old_counts.items():
            name = "Normal" if val == 0 else "Congested (binary)"
            print(f"    {val} ({name:25s}): {cnt:5d}  ({cnt/total:.1%})")

    # ── Filter to observable bins ───────────────────────────────
    if "is_observable" in df.columns:
        obs_mask = df["is_observable"] == 1
    else:
        obs_mask = pd.Series([True] * len(df))

    # Only cluster bins where all 3 features are present
    feat_mask = obs_mask & df[FEATURES].notna().all(axis=1)
    df_obs    = df[feat_mask].copy()
    print(f"\n  Observable bins with full host features: {len(df_obs)}")

    if len(df_obs) < K * 20:
        print("  ERROR: Not enough observable bins for K-Means. Requires at least",
              K * 20, "rows.")
        sys.exit(1)

    # ── Scale and cluster ───────────────────────────────────────
    X_raw    = df_obs[FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"\n  Running K-Means (K={K}, n_init=10, random_state={RANDOM_STATE})…")
    km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE)
    cluster_ids = km.fit_predict(X_scaled)

    # ── Cluster quality metrics ─────────────────────────────────
    sil = silhouette_score(X_scaled, cluster_ids, sample_size=5000,
                           random_state=RANDOM_STATE)
    ch  = calinski_harabasz_score(X_scaled, cluster_ids)
    db  = davies_bouldin_score(X_scaled, cluster_ids)

    print(f"\n  Cluster Quality:")
    print(f"    Silhouette Score          : {sil:.4f}  (higher better, >0.3 = meaningful)")
    print(f"    Calinski-Harabasz Index   : {ch:.1f}  (higher better)")
    print(f"    Davies-Bouldin Index      : {db:.4f}  (lower better, <1.0 = good)")

    # ── Map cluster IDs → label names ──────────────────────────
    label_map, centroids, order = map_clusters_to_labels(km, scaler, X_scaled)

    print(f"\n  Cluster Centroids (un-scaled) and Assignments:")
    print(f"  {'Cluster':>7} {'Label':>12} {'Binary':>6}  "
          f"{'RTT_ratio':>10} {'Jitter_us':>10} {'BW_ratio':>9}")
    for cid in range(K):
        name, binary = label_map[cid]
        c = centroids[cid]
        print(f"  {cid:>7} {name:>12} {binary:>6}  "
              f"{c[0]:>10.4f} {c[1]:>10.1f} {c[2]:>9.4f}")

    # ── Assign labels to observable rows ───────────────────────
    label_names_arr  = np.array([label_map[c][0] for c in cluster_ids])
    label_binary_arr = np.array([label_map[c][1] for c in cluster_ids])

    # ── Write back into full dataframe ─────────────────────────
    # Default non-observable rows to Normal (no host data = unknown, assume normal)
    df["congestion_label"]  = "Normal"
    df["congestion_binary"] = 0

    df.loc[feat_mask, "congestion_label"]  = label_names_arr
    df.loc[feat_mask, "congestion_binary"] = label_binary_arr

    # Preserve 3-class state column if it existed
    state_map = {"Normal": 0, "Onset": 1, "Congested": 2}
    df["congestion_state"] = df["congestion_label"].map(state_map)

    # ── New label distribution ──────────────────────────────────
    new_counts = df["congestion_label"].value_counts()
    binary_counts = df["congestion_binary"].value_counts().sort_index()
    total = len(df)

    print(f"\n  New label distribution (K-Means):")
    for label in ["Normal", "Onset", "Congested"]:
        cnt = new_counts.get(label, 0)
        print(f"    {label:12s}: {cnt:5d}  ({cnt/total:.1%})")
    print(f"\n  Binary:")
    for val, cnt in binary_counts.items():
        name = "Normal" if val == 0 else "Congested (onset+cong)"
        print(f"    {val} ({name}): {cnt:5d}  ({cnt/total:.1%})")

    # ── Save labeled dataset ────────────────────────────────────
    df.to_csv(DATASET_OUT, index=False)
    print(f"\n  Saved: {DATASET_OUT}")

    # ── Write report ────────────────────────────────────────────
    report = f"""K-Means Congestion Labeling Report
===================================
Input rows         : {len(df)}
Observable rows    : {len(df_obs)}
K                  : {K}
Clustering features: {', '.join(FEATURES)}
Random state       : {RANDOM_STATE}

Cluster Quality
---------------
Silhouette Score          : {sil:.4f}
Calinski-Harabasz Index   : {ch:.1f}
Davies-Bouldin Index      : {db:.4f}

Cluster → Label Mapping (by RTT ratio ascending)
-------------------------------------------------
"""
    for cid in range(K):
        name, binary = label_map[cid]
        c = centroids[cid]
        report += (f"  Cluster {cid} → {name:12s} (binary={binary}) | "
                   f"RTT_ratio={c[0]:.4f}, Jitter={c[1]:.0f}us, BW_ratio={c[2]:.4f}\n")

    report += "\nLabel Distribution\n------------------\n"
    for label in ["Normal", "Onset", "Congested"]:
        cnt = new_counts.get(label, 0)
        report += f"  {label:12s}: {cnt:5d}  ({cnt/total:.1%})\n"
    report += f"\n  Binary 0 (Normal)    : {binary_counts.get(0, 0):5d}\n"
    report += f"  Binary 1 (Congested) : {binary_counts.get(1, 0):5d}\n"

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report:  {REPORT_OUT}")

    print("\n  Next steps:")
    print("   1. Retrain demo model:   python3 demo/train_demo_model.py")
    print("   2. Retrain full models:  python3 models/train_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
