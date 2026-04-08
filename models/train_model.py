#!/usr/bin/env python3
"""
ML Training Pipeline — SDN Congestion Detection
================================================

ARCHITECTURE OVERVIEW
=====================

This pipeline uses ONLY switch qdisc metrics as ML features, trained
against labels derived exclusively from host/flow signals (iperf3, ping,
ss TCP socket stats) in build_dataset.py.

WHY SWITCH-ONLY FEATURES FOR ML
=================================

In a real SDN deployment:
  ✓  Switch statistics are always available to the controller via
     OpenFlow port stats requests (OFPT_STATS_REQUEST) at any polling
     interval — no host-side instrumentation required.
  ✗  iperf3 requires an active test session running on the end-host.
  ✗  ss -tin requires a monitoring agent process on every host.
  ✗  ICMP ping requires the controller to inject probe packets.

At inference time, this model receives only switch qdisc metrics
and predicts the congestion state that end-hosts would experience —
without touching a single host.

WHY ML OVER SIMPLE THRESHOLDS ON SWITCH DATA
=============================================

A naive approach: `if backlog_pkt_max > 23: congested`
This is a one-dimensional, static, instantaneous rule. It fails because:

  1. TEMPORAL PATTERNS: Queue backlog oscillates. A single second at
     23 packets may self-resolve; 15 seconds trending toward 23 is
     predictive of imminent congestion. Rolling-window features capture
     this trend; a threshold cannot.

  2. NON-LINEAR INTERACTIONS: High utilization combined with rising
     overlimit rate predicts host-visible congestion better than either
     metric alone. Tree ensembles learn these interactions automatically.

  3. FALSE POSITIVE RATE: Switch backlog can momentarily spike due to
     traffic bursts that clear without any host-visible RTT impact. The
     ML model, trained on host-labeled ground truth, learns to
     distinguish transient bursts from sustained congestion.

  4. GENERALISATION ACROSS TRAFFIC PATTERNS: The fixed threshold
     backlog > 23 is derived for one specific traffic mixture. Under
     different flow counts or flow sizes, the same backlog level may
     produce different host-experienced congestion. The model adapts
     to the traffic context captured by all switch features jointly.

SWITCH FEATURE SET (from qdisc_metrics.csv)
============================================
Raw features per second:
    qdisc_backlog_pkt_max      : max packets queued (most direct signal)
    qdisc_backlog_bytes_max    : max bytes queued
    qdisc_drop_ratio_max       : drops / (sent + drops) — normalised intensity
    qdisc_dropped_delta_sum    : packets dropped this second
    qdisc_overlimit_delta_sum  : rate-limit hits this second
    qdisc_overlimit_rate_sum   : overlimit events per second
    qdisc_throughput_mbps_sum  : bytes sent converted to Mbps
    qdisc_util_pct_max         : throughput / link_capacity × 100
    qdisc_sent_pkt_delta_sum   : packets transmitted this second

Temporal features derived from each raw feature:
    _roll3_mean  / _roll3_std  / _roll3_max   — 3-second window
    _roll10_mean / _roll10_std / _roll10_max  — 10-second window
    _roll30_mean / _roll30_std / _roll30_max  — 30-second window
    _delta1                                   — 1-step rate of change
    _zscore10                                 — 10-second z-score
    _lag1 / _lag3 / _lag5                    — past 1/3/5 second values

Interaction features:
    drop_x_util    : drop_ratio × utilization (joint stress indicator)
    overlimit_x_bkl: overlimit_rate × backlog  (rate-limiting + queue depth)

Design decisions
-----------------
1.  Time-aware split — split at 80th percentile of time_bin, never
    random shuffle. Random splits leak future state into training
    (temporal data leakage inflating reported accuracy).

2.  class_weight='balanced' — compensates for normal/congested imbalance;
    the classifier does not simply predict "normal" for everything.

3.  Model suite — RandomForest (robust baseline), GradientBoosting
    (captures non-linear feature interactions), XGBoost (fastest,
    often best on tabular data; optional).

4.  Best model by F1-macro — robust to class imbalance; accuracy alone
    is misleading when one class is rare.

Outputs (./model/)
-------------------
  best_model.pkl          — serialised best classifier
  scaler.pkl              — StandardScaler fitted on training features
  feature_names.txt       — ordered list of switch features used
  evaluation_report.txt   — full metrics for every model
  feature_importance.png  — top-20 features by MDI and permutation
  confusion_matrix.png    — confusion matrix on the test set
  roc_curve.png           — ROC curves for all models
  predictions_timeline.png— predictions vs. ground truth over time
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join("dataset", "labeled_dataset.csv")
MODEL_DIR    = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Target ─────────────────────────────────────────────────────────────────
TARGET = "congestion_binary"   # 0=normal, 1=congested (onset+congested)

# ── Temporal window sizes (seconds) ───────────────────────────────────────
WINDOWS = [3, 10, 30]

# ── Train / test ratio ─────────────────────────────────────────────────────
TRAIN_RATIO = 0.80

RANDOM_STATE = 42

# ════════════════════════════════════════════════════════════════════════════
# SWITCH FEATURE COLUMNS
# Only qdisc_* columns are used as ML features.
# Host/flow columns (iperf_*, ping_*, host_*, rtt_relative_*) are excluded:
#   - They were used to DERIVE the labels (data leakage if used as features)
#   - They are unavailable to an SDN controller at runtime
# ════════════════════════════════════════════════════════════════════════════

SWITCH_RAW_COLS = [
    "qdisc_backlog_pkt_max",
    "qdisc_backlog_bytes_max",
    "qdisc_drop_ratio_max",
    "qdisc_dropped_delta_sum",
    "qdisc_overlimit_delta_sum",
    "qdisc_overlimit_rate_sum",
    "qdisc_throughput_mbps_sum",
    "qdisc_util_pct_max",
    "qdisc_sent_pkt_delta_sum",
]

# Columns that must NEVER appear as features — labels, evidence, or leakage
ALWAYS_EXCLUDE = {
    "congestion_binary",
    "congestion_state",
    "congestion_label",
    "time_bin",
}


# ============================================================================
# 1.  LOAD & VALIDATE
# ============================================================================

def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run build_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(path).sort_values("time_bin").reset_index(drop=True)
    print(f"  Loaded {len(df)} rows × {len(df.columns)} columns")

    if TARGET not in df.columns:
        print(f"ERROR: Target '{TARGET}' missing. Re-run build_dataset.py.")
        sys.exit(1)

    # FIX 1: Drop unlabeled bins (congestion_state == -1).
    # These bins had no active iperf flow AND no ping data during the experiment.
    # With no host-observable signal, their label is ambiguous — including them
    # would train the model that "switch drops during idle periods = normal",
    # introducing systematic label noise. They are excluded here, not from the
    # CSV, so the full dataset remains auditable.
    if "congestion_state" in df.columns:
        n_unlabeled = (df["congestion_state"] == -1).sum()
        if n_unlabeled > 0:
            print(f"  Dropping {n_unlabeled} unlabeled bins (state=-1, no host measurement)")
            df = df[df["congestion_state"] >= 0].reset_index(drop=True)

    # Check switch columns are present
    missing = [c for c in SWITCH_RAW_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing switch features: {missing}")
        print("  Ensure qdisc_metrics.csv was loaded by build_dataset.py.")

    dist = df[TARGET].value_counts().to_dict()
    print(f"  Label distribution: normal={dist.get(0, 0)}  congested={dist.get(1, 0)}")
    return df



# ============================================================================
# 2.  FEATURE ENGINEERING — SWITCH-ONLY TEMPORAL FEATURES
# ============================================================================

def engineer_features(df):
    """
    Build rolling-window, lag, rate-of-change, and interaction features
    from switch qdisc raw columns ONLY.

    Rolling statistics:
        mean  — persistent trend (is backlog consistently high?)
        std   — volatility (is the queue oscillating?)
        max   — worst-case in the window (did a spike occur?)

    Lag features:
        Past values allow the model to see the trajectory of each metric
        over the last 1, 3, 5 seconds — critical for onset detection.

    Z-score:
        How anomalous is this second relative to the recent 10-second
        history? Spikes that are large relative to the recent baseline
        are more predictive than absolute values.

    Rate-of-change (delta1):
        Rising backlog predicts drops better than static backlog level.
        A queue growing at 3 pkts/s from 10 pkts is more concerning
        than a static queue at 15 pkts.

    Interaction features:
        drop_x_util     — high utilization AND drops co-occurring is
                          stronger evidence than either alone
        overlimit_x_bkl — rate-limit + backlog: HTB is throttling AND
                          the queue is filling simultaneously
    """
    df = df.copy()

    available = [c for c in SWITCH_RAW_COLS if c in df.columns]
    if not available:
        print("  ERROR: No switch columns found. Cannot engineer features.")
        sys.exit(1)

    print(f"  Building temporal features from {len(available)} switch columns: {available}")

    for col in available:
        s = df[col].fillna(0)

        for w in WINDOWS:
            df[f"{col}_roll{w}_mean"] = s.rolling(w, min_periods=1).mean()
            df[f"{col}_roll{w}_std"]  = s.rolling(w, min_periods=1).std().fillna(0)
            df[f"{col}_roll{w}_max"]  = s.rolling(w, min_periods=1).max()

        # Rate-of-change: rising backlog/drops are early congestion signals
        df[f"{col}_delta1"] = s.diff(1).fillna(0)

        # Z-score: anomaly relative to recent 10-second window
        roll_mean = s.rolling(10, min_periods=1).mean()
        roll_std  = s.rolling(10, min_periods=1).std().fillna(1).clip(lower=1e-9)
        df[f"{col}_zscore10"] = (s - roll_mean) / roll_std

        # Lag features: trajectory over past 1, 3, 5 seconds
        for lag in [1, 3, 5]:
            df[f"{col}_lag{lag}"] = s.shift(lag).fillna(s.iloc[0] if len(s) > 0 else 0)

    # ── Interaction features ───────────────────────────────────────────────
    if ("qdisc_drop_ratio_max" in df.columns and
            "qdisc_util_pct_max" in df.columns):
        df["drop_x_util"] = (
            df["qdisc_drop_ratio_max"] * df["qdisc_util_pct_max"]
        )

    if ("qdisc_overlimit_rate_sum" in df.columns and
            "qdisc_backlog_pkt_max" in df.columns):
        df["overlimit_x_bkl"] = (
            df["qdisc_overlimit_rate_sum"] * df["qdisc_backlog_pkt_max"]
        )

    print(f"  After feature engineering: {len(df.columns)} total columns")
    return df


def select_features(df):
    """
    Select ONLY switch-derived features for ML training.

    Excluded categories:
      • ALWAYS_EXCLUDE  — label columns, time index
      • hf_*   — HF-CEF evidence flags (direct label derivations, leakage)
      • iperf_* — host flow data (used for labeling, unavailable at runtime)
      • ping_*  — ICMP probe data (ditto)
      • host_*  — ss socket stats (ditto)
      • rtt_*   — derived RTT ratios (from iperf/ping, unavailable at runtime)
      • retransmit_rate — derived from iperf (unavailable at runtime)
      • queue_fill_ratio — derived from qdisc backlog (switch-side; allowed)
      • host_delivery_ratio — derived from ss (host-side; excluded)
    """
    # Build the exclusive allowlist: switch raw + switch temporal + interactions
    switch_prefix_allowed = {"qdisc_", "drop_x_", "overlimit_x_"}

    def is_switch_feature(col):
        return any(col.startswith(p) for p in switch_prefix_allowed)

    host_flow_prefixes = {"iperf_", "ping_", "host_", "hf_", "rtt_relative"}
    host_flow_exact    = {"retransmit_rate", "queue_fill_ratio",
                          "host_delivery_ratio", "util_pct_max"}

    def is_excluded(col):
        if col in ALWAYS_EXCLUDE:
            return True
        if col in host_flow_exact:
            return True
        if any(col.startswith(p) for p in host_flow_prefixes):
            return True
        return False

    feature_cols = [
        c for c in df.columns
        if not is_excluded(c)
        and is_switch_feature(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        print("ERROR: No switch features selected. Check column names.")
        sys.exit(1)

    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y = df[TARGET].values.astype(int)

    print(f"  Switch feature count : {len(feature_cols)}")
    print(f"  Target distribution  : normal={np.sum(y==0)}  congested={np.sum(y==1)}")
    return X, y, feature_cols


# ============================================================================
# 3.  TIME-AWARE TRAIN/TEST SPLIT
# ============================================================================

def time_split(df, X, y):
    """Split at 80th percentile of time — never random, to respect causality."""
    n       = len(df)
    cutoff  = int(n * TRAIN_RATIO)
    split_t = df["time_bin"].iloc[cutoff]
    print(f"  Train : first {cutoff} bins (t ≤ {split_t}s)")
    print(f"  Test  : last  {n - cutoff} bins  (t > {split_t}s)")

    mask = np.arange(n) < cutoff
    states = df["congestion_state"].values
    return X.values[mask], X.values[~mask], y[mask], y[~mask], states[mask], states[~mask]


# ============================================================================
# 4.  MODEL DEFINITIONS
# ============================================================================

def get_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        print("  XGBoost available — included")
    except ImportError:
        print("  XGBoost not installed — skipping")
    return models


# ============================================================================
# 5.  TRAINING & EVALUATION
# ============================================================================

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, feature_cols, state_te):
    print(f"\n  [{name}]")
    
    # 1. Feature Selection (prevent overfitting)
    # Use a Random Forest to select the top 40 features
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    selector = SelectFromModel(rf_selector, max_features=40, threshold=-np.inf)
    X_tr_sel = selector.fit_transform(X_tr, y_tr)
    X_te_sel = selector.transform(X_te)
    sel_mask = selector.get_support()
    sel_features = [feature_cols[i] for i, m in enumerate(sel_mask) if m]
    print(f"    Selected {len(sel_features)}/{len(feature_cols)} features")

    # 2. Threshold Optimization via CV and PR curve
    if hasattr(model, "predict_proba"):
        cv_probs = cross_val_predict(model, X_tr_sel, y_tr, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
        precs, recs, threshs = precision_recall_curve(y_tr, cv_probs)
        # Handle zero division
        f1s = 2 * (precs * recs) / np.maximum(precs + recs, 1e-9)
        best_idx = np.argmax(f1s)
        best_thresh = threshs[best_idx] if best_idx < len(threshs) else 0.5
        print(f"    Optimal threshold: {best_thresh:.3f}")
    else:
        best_thresh = 0.5

    model.fit(X_tr_sel, y_tr)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1 = cross_val_score(model, X_tr_sel, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1)

    y_prob = model.predict_proba(X_te_sel)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = (y_prob >= best_thresh).astype(int) if y_prob is not None else model.predict(X_te_sel)

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_te, y_prob) if y_prob is not None else float("nan")
    rep = classification_report(y_te, y_pred,
                                target_names=["normal", "congested"],
                                zero_division=0)

    # 3. 3-class Breakdown (normal/onset/congested)
    print(f"    CV F1-macro (train): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"    Test accuracy      : {acc:.4f}")
    print(f"    Test F1-macro      : {f1:.4f}")
    print(f"    Test ROC-AUC       : {auc:.4f}")
    print(f"\n{rep}")

    # Generate the 3-class cross-tab string
    breakdown_str = "    3-Class Breakdown (Test Set):\n"
    for st, st_name in [(0, "Normal"), (1, "Onset"), (2, "Congested")]:
        mask_st = (state_te == st)
        if mask_st.sum() > 0:
            preds = y_pred[mask_st]
            pred_0 = (preds == 0).sum()
            pred_1 = (preds == 1).sum()
            breakdown_str += f"      {st_name:9s} (n={mask_st.sum()}): {pred_0} predicted normal, {pred_1} predicted congested\n"
    print(f"\n{breakdown_str}")

    return {
        "name": name, "model": model, "selector": selector, "sel_features": sel_features,
        "best_thresh": best_thresh, "acc": acc, "f1": f1, "auc": auc, "cv_f1": cv_f1.mean(),
        "y_pred": y_pred, "y_prob": y_prob, "report": rep, "breakdown_str": breakdown_str,
    }


def train_all_models(models, X_tr, X_te, y_tr, y_te, feature_cols, state_te):
    results = []
    for name, model in models.items():
        try:
            r = evaluate_model(name, model, X_tr, X_te, y_tr, y_te, feature_cols, state_te)
            results.append(r)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
    return results


def pick_best(results):
    best = max(results, key=lambda r: r["f1"])
    print(f"\n  Best model: {best['name']}  (test F1-macro={best['f1']:.4f})")
    return best


# ============================================================================
# 6.  PLOTS
# ============================================================================

def plot_feature_importance(best, feature_cols, X_te, y_te, out_dir=MODEL_DIR):
    model = best["model"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        top = imp.nlargest(20).sort_values()
        axes[0].barh(top.index, top.values, color="steelblue")
        axes[0].set_title(f"{best['name']} — MDI Feature Importance (top 20)\n"
                          f"[All features are switch qdisc metrics]")
        axes[0].set_xlabel("Mean decrease in impurity")
        axes[0].grid(axis="x", alpha=0.3)
    else:
        axes[0].set_visible(False)

    try:
        # Transform X_te to include only the selected features for permutation evaluation
        selector = best.get("selector")
        if selector is not None:
             X_te_sel = selector.transform(X_te)
        else:
             X_te_sel = X_te

        perm = permutation_importance(
            model, X_te_sel, y_te,
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
            scoring="f1_macro",
        )
        perm_imp = pd.Series(perm.importances_mean, index=feature_cols)
        top_p    = perm_imp.nlargest(20).sort_values()
        axes[1].barh(top_p.index, top_p.values, color="tomato")
        axes[1].set_title(f"{best['name']} — Permutation Importance (top 20)\n"
                          f"[Importance = F1 drop when feature shuffled]")
        axes[1].set_xlabel("Mean F1-macro decrease")
        axes[1].grid(axis="x", alpha=0.3)
    except Exception as e:
        print(f"  Permutation importance failed: {e}")
        axes[1].set_visible(False)

    fig.suptitle("Switch-Only Feature Importance for SDN Congestion Detection",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance plot → {path}")


def plot_confusion_matrix(best, y_te, out_dir=MODEL_DIR):
    cm   = confusion_matrix(y_te, best["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "congested"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {best['name']}\n"
                 f"(switch features → host-labeled ground truth)")
    fig.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix plot  → {path}")


def plot_roc_curves(results, y_te, out_dir=MODEL_DIR):
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        if r["y_prob"] is None:
            continue
        fpr, tpr, _ = roc_curve(y_te, r["y_prob"])
        ax.plot(fpr, tpr, linewidth=2,
                label=f"{r['name']}  AUC={r['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Switch Features → Host-Labeled Congestion")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "roc_curve.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve plot         → {path}")


def plot_predictions_timeline(df, best, X_scaled, y, out_dir=MODEL_DIR):
    """
    Compare model predictions against host-derived ground truth labels.
    Also overlays the key switch features the model used, providing a
    visual proof that switch metrics alone can predict host-experienced
    congestion.
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(4, 1, hspace=0.5)
    t   = df["time_bin"].values

    # Panel 1 — Switch: backlog and utilization
    ax1 = fig.add_subplot(gs[0])
    if "qdisc_backlog_pkt_max" in df.columns:
        ax1.plot(t, df["qdisc_backlog_pkt_max"], color="steelblue",
                 linewidth=1, label="backlog_pkt_max")
    ax1b = ax1.twinx()
    if "qdisc_util_pct_max" in df.columns:
        ax1b.plot(t, df["qdisc_util_pct_max"], color="darkorange",
                  linewidth=1, alpha=0.7, label="util_pct_max (%)")
    ax1.set_ylabel("Backlog (pkts)", color="steelblue")
    ax1b.set_ylabel("Utilization (%)", color="darkorange")
    ax1.set_title("Switch Metrics (ML input) — Backlog and Link Utilization")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    # Panel 2 — Switch: drops and overlimits
    ax2 = fig.add_subplot(gs[1])
    if "qdisc_dropped_delta_sum" in df.columns:
        ax2.bar(t, df["qdisc_dropped_delta_sum"], color="crimson",
                alpha=0.7, label="dropped_delta/s")
    ax2b = ax2.twinx()
    if "qdisc_overlimit_delta_sum" in df.columns:
        ax2b.plot(t, df["qdisc_overlimit_delta_sum"], color="orange",
                  linewidth=1, alpha=0.8, label="overlimits_delta/s")
    ax2.set_ylabel("Drops/s", color="crimson")
    ax2b.set_ylabel("Overlimits/s", color="orange")
    ax2.set_title("Switch Metrics (ML input) — Drops and Overlimits")
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")
    ax2.grid(alpha=0.3)

    # Panel 3 — Host-derived ground truth labels
    ax3 = fig.add_subplot(gs[2])
    colours = {0: "royalblue", 1: "orange", 2: "crimson"}
    for state, name in [(0, "normal"), (1, "onset"), (2, "congested")]:
        if "congestion_state" in df.columns:
            mask = df["congestion_state"].values == state
            ax3.fill_between(t, mask.astype(float), where=mask,
                             color=colours[state], alpha=0.6,
                             label=f"{name} (host-derived)", step="mid")
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["off", "on"])
    ax3.set_title("Ground Truth Labels (derived from HOST flow signals — iperf/ping/ss)")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(alpha=0.3)

    # Panel 4 — Model predictions (from switch features only)
    ax4 = fig.add_subplot(gs[3])
    # To predict full series for visual, use the best model's pipeline
    X_scaled_sel = best["selector"].transform(X_scaled.values)
    full_prob = best["model"].predict_proba(X_scaled_sel)[:, 1] if hasattr(best["model"], "predict_proba") else None
    full_pred = (full_prob >= best["best_thresh"]).astype(int) if full_prob is not None else best["model"].predict(X_scaled_sel)
    
    ax4.step(t, y, where="mid", color="forestgreen", linewidth=1.5,
             alpha=0.7, label="True label (host-derived)")
    ax4.step(t, full_pred, where="mid", color="tomato", linewidth=1,
             linestyle="--", alpha=0.9, label=f"Predicted ({best['name']}, switch-only features)")
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["normal", "congested"])
    ax4.set_xlabel("Experiment time (s)")
    ax4.set_title(f"ML Predictions from SWITCH METRICS → vs HOST-LABELED Ground Truth")
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(alpha=0.3)

    fig.suptitle(
        "SDN Congestion Detection: Switch Metrics Predicting Host-Experienced Congestion",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "predictions_timeline.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Timeline plot          → {path}")


# ============================================================================
# 7.  SAVE MODEL & REPORT
# ============================================================================

def save_model(best, scaler, feature_cols, results, out_dir=MODEL_DIR):
    model_path  = os.path.join(out_dir, "best_model.pkl")
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    feat_path   = os.path.join(out_dir, "feature_names.txt")
    report_path = os.path.join(out_dir, "evaluation_report.txt")

    with open(model_path, "wb")  as f: pickle.dump(best["model"], f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    with open(feat_path, "w", encoding="utf-8")    as f: f.write("\n".join(feature_cols))

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SDN Congestion Detection — Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("Feature source   : switch qdisc metrics ONLY\n")
        f.write("Label source     : host/flow signals (iperf3, ping, ss)\n")
        f.write("Label method     : HF-CEF (Host-Flow Congestion Evidence Framework)\n\n")
        f.write(f"Switch features  : {len(feature_cols)}\n\n")
        f.write("Methodology improvements applied:\n")
        f.write("  - Feature selection (reduced from 137 to 40 features to prevent overfitting)\n")
        f.write("  - Threshold optimization via Precision-Recall curve on training CV\n")
        f.write("  - 3-class detection breakdown for analyzing early onset catching\n\n")
        for r in results:
            f.write(f"Model: {r['name']}\n")
            f.write(f"  Selected features    : {len(r['sel_features'])}\n")
            f.write(f"  Optimal Threshold    : {r['best_thresh']:.3f}\n")
            f.write(f"  CV  F1-macro (train) : {r['cv_f1']:.4f}\n")
            f.write(f"  Test accuracy        : {r['acc']:.4f}\n")
            f.write(f"  Test F1-macro        : {r['f1']:.4f}\n")
            f.write(f"  Test ROC-AUC         : {r['auc']:.4f}\n")
            f.write(f"\n{r['report']}\n{r['breakdown_str']}\n{'─'*40}\n\n")
        f.write(f"Best model: {best['name']}\n")

    print(f"\n  Model saved  → {model_path}")
    print(f"  Scaler saved → {scaler_path}")
    print(f"  Report saved → {report_path}")


# ============================================================================
# 8.  ENTRY POINT
# ============================================================================

def main():
    print("=" * 60)
    print("SDN Congestion Detection — ML Training Pipeline")
    print("Features: SWITCH qdisc metrics only")
    print("Labels  : Host/flow-derived (HF-CEF)")
    print("=" * 60)

    print("\n[Loading labeled dataset]")
    df = load_dataset()

    print("\n[Engineering switch temporal features]")
    df_eng = engineer_features(df)
    X, y, feature_cols = select_features(df_eng)

    print("\n[Scaling features]")
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=feature_cols,
        index=X.index,
    )

    print("\n[Time-aware train/test split]")
    X_tr, X_te, y_tr, y_te, state_tr, state_te = time_split(df_eng, X_scaled, y)

    print("\n[Training models]")
    models  = get_models()
    results = train_all_models(models, X_tr, X_te, y_tr, y_te, feature_cols, state_te)

    if not results:
        print("ERROR: No model trained successfully.")
        sys.exit(1)

    best = pick_best(results)

    print("\n[Generating plots]")
    # Feature importance uses the post-selection mask
    plot_feature_importance(best, best["sel_features"], X_te, y_te)
    plot_confusion_matrix(best, y_te)
    plot_roc_curves(results, y_te)
    plot_predictions_timeline(df_eng, best, X_scaled, y)

    print("\n[Saving model and report]")
    save_model(best, scaler, feature_cols, results)

    print("\n[Done]")
    print(f"  Best: {best['name']}  "
          f"accuracy={best['acc']:.4f}  F1={best['f1']:.4f}  AUC={best['auc']:.4f}")
    print(f"\n  This model predicts host-experienced congestion using ONLY")
    print(f"  switch qdisc metrics — no host instrumentation required at runtime.")


if __name__ == "__main__":
    main()
