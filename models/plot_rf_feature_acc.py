"""
Plot: Number of Top RF Features vs. Accuracy (and F1)
Uses the SAME feature engineering as train_model.py so results are consistent.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

# ── reproduce the same feature pipeline from train_model.py ────────────────
LABEL_COL   = "congestion_binary"
RAW_QDISC   = ["qdisc_backlog_pkt_max", "qdisc_backlog_bytes_max",
               "qdisc_drop_ratio_max", "qdisc_dropped_delta_sum",
               "qdisc_overlimit_delta_sum", "qdisc_overlimit_rate_sum",
               "qdisc_throughput_mbps_sum", "qdisc_util_pct_max",
               "qdisc_sent_pkt_delta_sum"]

def engineer_features(df):
    """Mirror the feature_engineering() from train_model.py."""
    base_cols = [c for c in RAW_QDISC if c in df.columns]
    for col in base_cols:
        s = df[col]
        for w in [10, 30]:
            df[f"{col}_roll{w}_mean"] = s.rolling(w, min_periods=1).mean()
            df[f"{col}_roll{w}_std"]  = s.rolling(w, min_periods=1).std().fillna(0)
            df[f"{col}_roll{w}_max"]  = s.rolling(w, min_periods=1).max()
        df[f"{col}_delta1"]    = s.diff(1).fillna(0)
        roll_mean = s.rolling(10, min_periods=1).mean()
        roll_std  = s.rolling(10, min_periods=1).std().fillna(1).clip(lower=1e-9)
        df[f"{col}_zscore10"]  = (s - roll_mean) / roll_std
        for lag in [1, 3, 5]:
            df[f"{col}_lag{lag}"] = s.shift(lag).fillna(s.iloc[0] if len(s) > 0 else 0)

    if "qdisc_drop_ratio_max" in df.columns and "qdisc_util_pct_max" in df.columns:
        df["drop_x_util"] = df["qdisc_drop_ratio_max"] * df["qdisc_util_pct_max"]
    if "qdisc_overlimit_rate_sum" in df.columns and "qdisc_backlog_pkt_max" in df.columns:
        df["overlimit_x_bkl"] = df["qdisc_overlimit_rate_sum"] * df["qdisc_backlog_pkt_max"]
    return df

def get_switch_features(df):
    allowed = {"qdisc_", "drop_x_", "overlimit_x_"}
    return [c for c in df.columns
            if any(c.startswith(p) for p in allowed)
            and pd.api.types.is_numeric_dtype(df[c])
            and c != LABEL_COL]

def main():
    print("=" * 55)
    print(" Random Forest: No. of Features vs. Accuracy / F1")
    print("=" * 55)

    # ── Load and engineer ────────────────────────────────────
    print("Loading dataset...")
    df = pd.read_csv("dataset/labeled_dataset.csv")
    df = engineer_features(df)
    switch_cols = get_switch_features(df)
    print(f"  Switch feature pool : {len(switch_cols)} columns")

    df = df.dropna(subset=switch_cols + [LABEL_COL])
    X = df[switch_cols].values
    y = df[LABEL_COL].values
    print(f"  Samples after dropna: {len(y)}")

    # ── Stratified 80/20 split ───────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ── Rank features by importance ──────────────────────────
    print("Fitting base RF to rank features...")
    rf_base = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    rank_order = np.argsort(rf_base.feature_importances_)[::-1]

    # ── Sweep k values ───────────────────────────────────────
    # Fine-grained 1-20, then coarser steps up to full feature pool
    k_values = (list(range(1, 21))
                + list(range(25, min(len(switch_cols), 80), 5))
                + [len(switch_cols)])
    k_values = sorted(set(k_values))

    accuracies, f1s = [], []

    print(f"Sweeping k from 1 to {k_values[-1]}…")
    for k in k_values:
        top_k = rank_order[:k]
        rf_k  = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        rf_k.fit(X_train[:, top_k], y_train)
        preds = rf_k.predict(X_test[:, top_k])
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="macro")
        accuracies.append(acc)
        f1s.append(f1)
        print(f"  k={k:3d} | Acc={acc:.4f} | F1={f1:.4f}")

    # ── Plot ─────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)
    out_path = os.path.join("model", "rf_features_vs_accuracy.png")

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Accuracy (left y-axis)
    ax1.plot(k_values, accuracies, "o-", color="#1f77b4", linewidth=2,
             markersize=5, label="Test Accuracy")
    ax1.set_xlabel("Number of Top Features Selected", fontsize=12)
    ax1.set_ylabel("Accuracy", color="#1f77b4", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax1.set_ylim([max(0.0, min(accuracies) - 0.04), 1.001])
    ax1.grid(True, linestyle="--", alpha=0.5)

    # F1 (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(k_values, f1s, "s--", color="#d62728", linewidth=2,
             markersize=5, label="Macro F1-Score")
    ax2.set_ylabel("Macro F1-Score", color="#d62728", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax2.set_ylim([max(0.0, min(f1s) - 0.04), 1.001])

    # Mark the elbow (best F1)
    best_k  = k_values[int(np.argmax(f1s))]
    best_f1 = max(f1s)
    ax2.axvline(best_k, color="grey", linestyle=":", linewidth=1.5)
    ax2.annotate(f"Best F1={best_f1:.4f}\n(k={best_k})",
                 xy=(best_k, best_f1),
                 xytext=(best_k + max(k_values) * 0.04, best_f1 - 0.025),
                 fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="grey"))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)

    plt.title("Random Forest — No. of Top Features vs. Accuracy & Macro F1",
              fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to: {out_path}")

if __name__ == "__main__":
    main()
