#!/usr/bin/env python3
"""
Train a lightweight demo-specific model on exactly the 15 selected features.
Outputs: model/demo_model.pkl  and  model/demo_scaler.pkl
Run once from the project root:
    python3 demo/train_demo_model.py
"""
import os, sys, pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

LABEL_COL  = "congestion_binary"
FEAT_FILE  = os.path.join(ROOT, "model", "feature_names.txt")
FEAT_NAMES = open(FEAT_FILE).read().splitlines()   # 15 features, in original model order

RAW_QDISC = [
    "qdisc_backlog_pkt_max", "qdisc_backlog_bytes_max",
    "qdisc_drop_ratio_max",  "qdisc_dropped_delta_sum",
    "qdisc_overlimit_delta_sum", "qdisc_overlimit_rate_sum",
    "qdisc_throughput_mbps_sum", "qdisc_util_pct_max",
    "qdisc_sent_pkt_delta_sum",
]

def engineer(df):
    new_cols = {}
    for col in [c for c in RAW_QDISC if c in df.columns]:
        s = df[col]
        for w in [10, 30]:
            new_cols[f"{col}_roll{w}_mean"] = s.rolling(w, min_periods=1).mean()
            new_cols[f"{col}_roll{w}_max"]  = s.rolling(w, min_periods=1).max()
    return df.assign(**new_cols)

if __name__ == "__main__":
    print("Loading dataset …")
    df = pd.read_csv(os.path.join(ROOT, "dataset", "labeled_dataset.csv"))
    df = engineer(df)

    missing = [f for f in FEAT_NAMES if f not in df.columns]
    if missing:
        print(f"ERROR: Missing engineered columns: {missing[:5]}"); sys.exit(1)

    df = df.dropna(subset=FEAT_NAMES + [LABEL_COL])
    X  = df[FEAT_NAMES].values        # exactly the 15 in the right order
    y  = df[LABEL_COL].values

    print(f"  {len(y)} samples | features: {X.shape[1]} | congestion rate: {y.mean():.1%}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(sss.split(X, y))
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_tr)
    X_te_sc   = scaler.transform(X_te)

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(X_tr_sc, y_tr)

    acc = accuracy_score(y_te, model.predict(X_te_sc))
    f1  = f1_score(y_te, model.predict(X_te_sc), average="macro")
    print(f"  Demo model: Acc={acc:.4f}  F1={f1:.4f}")

    pickle.dump(model,  open(os.path.join(ROOT, "model", "demo_model.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(ROOT, "model", "demo_scaler.pkl"), "wb"))
    # Also save feature order used
    with open(os.path.join(ROOT, "model", "demo_feature_names.txt"), "w") as f:
        f.write("\n".join(FEAT_NAMES))
    print("  Saved: model/demo_model.pkl  model/demo_scaler.pkl")
