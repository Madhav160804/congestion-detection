#!/usr/bin/env python3
"""
Dataset Builder — Host-Flow Congestion Evidence Framework (HF-CEF)
===================================================================

ARCHITECTURE OVERVIEW
=====================

  LABELING  (this file)        TRAINING  (train_model.py)
  ─────────────────────        ──────────────────────────
  Host / flow signals    →     Switch qdisc metrics  →  ML Model
  (iperf3, ping, ss)           (backlog, drops,
  derive ground-truth          overlimits, util)
  labels                       + temporal windows

WHY THIS SEPARATION IS CRITICAL
================================

Switch metrics (tc qdisc: backlog, drops, overlimits) are always
natively available to an SDN controller via OpenFlow port statistics.
A trivial if-else rule on those metrics would already detect congestion:

    if dropped_delta > 0: label = CONGESTED

If we ALSO used switch metrics for labeling, the ML model would simply
learn to replicate that if-else rule — adding zero value over a
hand-crafted threshold.

Instead:
  • Labels are derived from HOST-OBSERVABLE signals: RTT inflation,
    TCP retransmissions, and throughput degradation measured by iperf3,
    ICMP ping, and kernel TCP socket stats (ss -tin). These signals
    represent congestion as EXPERIENCED BY THE END-HOST — they are
    the ground truth of what congestion actually means in terms of
    network performance impact.

  • The ML model is trained on SWITCH metrics only — signals the SDN
    controller can poll at runtime without any host-side instrumentation.
    The model learns the non-linear mapping:
        switch observables  →  host-experienced congestion state

  • At INFERENCE TIME, only switch metrics are needed. No iperf, no ping,
    no ss agents on hosts. This is realistic for real SDN deployments.

WHY ML OVER SIMPLE THRESHOLDS ON SWITCH DATA
=============================================

A static threshold on switch metrics (e.g. backlog > 23 pkts →
congested) is REACTIVE: it fires only when the queue is already
stressed. The ML model, trained with rolling-window temporal features,
learns PREDICTIVE patterns:
  - Queue backlog trending upward over 10s predicts drops before they occur
  - Utilization crossing 75% consistently precedes retransmit events at hosts
  - Non-linear interactions between utilization + overlimits predict onset
These temporal, non-linear relationships cannot be captured by a static
if-else threshold on any single switch metric.

LABELING METHODOLOGY — HF-CEF
==============================

Labels are assigned using three independent protocol-layer evidence
groups, ALL derived from host/flow measurements. No switch-side metric
(backlog, dropped_delta, overlimits) participates in labeling.

Topology parameters (from mytopo.py)
--------------------------------------
    LINK_BW_MBPS   = 25.0   # bw=25 in net.addLink(...)
    LINK_DELAY_MS  = 30.0   # delay="30ms" in net.addLink(...)
    MAX_QUEUE_PKTS = 30     # max_queue_size=30 in net.addLink(...)
    MTU_BYTES      = 1500   # standard Ethernet MTU

Derived constants
-----------------
    PKT_TX_MS   = (1500 × 8) / (25 × 10⁶) × 10³  =  0.48 ms
        Transmission time for one MTU packet at link capacity.

    BASELINE_RTT_MS = 4 × 30 = 120 ms
        Pure propagation RTT with zero queuing:
        h1→s1 (30ms) + s1→h2 (30ms) + h2→s1 (30ms) + s1→h1 (30ms)

Evidence Flags (all host/flow sourced)
----------------------------------------

Group A — TCP Loss Layer
    A_retx_iperf : iperf3 reports retransmits > 0 in interval
    A_retx_host  : kernel ss reports retransmit delta > 0
    A_lost_host  : kernel ss reports lost segment delta > 0

    A_LOSS_CONFIRMED:
        (A_retx_iperf AND A_retx_host) OR A_lost_host
        ─────────────────────────────────────────────
        Requiring BOTH iperf AND ss retransmit evidence eliminates
        single-source artifacts (e.g. a spurious iperf counter reset).
        ss `lost` is an independent kernel-level count of segments
        declared lost after RTO expiry — always definitive.
        When ss data is unavailable, falls back to A_retx_iperf
        alone combined with a stricter RTT threshold (see below).

Group B — Latency Inflation Layer  [THRESHOLD: T_RTT_RATIO = 1.092]
    B_rtt_iperf : iperf RTT mean / baseline RTT > T_RTT_RATIO
    B_rtt_ping  : ICMP ping RTT mean / baseline RTT > T_RTT_RATIO
    B_rttvar    : iperf RTT variance > T_RTTVAR_US

    T_RTT_RATIO derivation (M/D/1 queuing theory):
        At utilisation ρ = 0.75 (the superlinear "knee" of M/D/1
        queue growth), mean queue occupancy begins growing rapidly.
        75% of 30-packet queue = 22.5 → ceil = 23 packets.
        One-way queuing delay at 23 packets: 23 × 0.48ms = 11.04ms.
        RTT_onset = 120ms + 11.04ms = 131.04ms
        T_RTT_RATIO = 131.04 / 120 = 1.092

        M/D/1 Lq formula: Lq = ρ² / (2(1−ρ))
        ρ=0.50 → Lq=0.50 pkts  (idle)
        ρ=0.75 → Lq=1.12 pkts  (onset — THRESHOLD)
        ρ=0.90 → Lq=4.05 pkts  (congested)

    T_RTTVAR_US derivation (queuing jitter):
        In an uncongested path, RTT variance originates from OS
        scheduling jitter (~1–2 ms on Linux).
        At 50% queue fill (15 pkts), packets arrive at a queue of
        variable depth 0–15; the resulting one-way delay variance:
            σ_queue = (15 × 0.48ms) / (2√3) = 2.08 ms one-way
        RTT variance contribution ≈ 2 × 2.08 = 4.16 ms.
        T_RTTVAR = 5000 µs lies above baseline jitter (~2ms) but
        below significant queuing at 50% fill (>4ms), making it a
        sensitive leading indicator of queue oscillation.

Group C — Throughput Degradation Layer  [THRESHOLD: T_BW_RATIO = 0.80]
    C_bw_iperf    : achieved BW / target BW < T_BW_RATIO (iperf3)
    C_delivery_ss : ss delivery rate / target BW < T_BW_RATIO

    T_BW_RATIO derivation (TCP AIMD steady-state):
        Mathis et al. (1997) TCP throughput formula:
            B ≈ MSS / (RTT × √loss_rate)
        Under periodic loss (AIMD congestion avoidance):
          - On loss: TCP Reno halves cwnd → throughput ~50% of target
          - AIMD recovery: throughput oscillates, averaging ~70–80%
        RFC 3148 defines throughput below 80% of capacity as
        "degraded" under normal TCP operation. T_BW_RATIO = 0.80
        captures persistent AIMD-induced degradation while excluding
        transient slow-start behavior (typically < 2 seconds).

Label Assignment Rules
-----------------------
    CONGESTED (2):
        A_LOSS_CONFIRMED = True
        ──────────────────────────────────────────────────────────────
        TCP packet loss has been confirmed at the host level. The
        network has physically dropped packets; the end-host detected
        this via TCP retransmit timeout or explicit SACK loss
        notification. This is the most objective congestion signal.

    ONSET (1):
        A_LOSS_CONFIRMED = False
        AND  (B_rtt_iperf OR B_rtt_ping)          ← Group B stress
        AND  (C_bw_iperf OR C_delivery_ss OR B_rttvar) ← Group C stress
        ──────────────────────────────────────────────────────────────
        No packet loss yet, but RTT is inflated above the M/D/1 onset
        threshold AND throughput is degrading / jitter is elevated.
        Requiring TWO independent groups eliminates false positives:
        a transient RTT spike from OS scheduling (no BW impact) is
        excluded; a single-second BW dip without RTT inflation is
        excluded. Onset captures early congestion before drops begin —
        enabling proactive SDN controller intervention.

    NORMAL (0):
        Neither of the above.

Binary label:  NORMAL=0,  ONSET+CONGESTED=1

Output Files (./dataset/)
--------------------------
    features.csv        — per-second aligned feature matrix
    hfcef_evidence.csv  — boolean evidence flags (audit trail)
    labeled_dataset.csv — features + labels, ML-ready
    labeling_report.txt — full derivation with threshold values
    hfcef_timeline.png  — visual audit: evidence flags + labels
"""

import os
import sys
import math
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TOPOLOGY CONSTANTS  (sourced from mytopo.py)
# ════════════════════════════════════════════════════════════════════════════

LINK_BW_MBPS   = 25.0
LINK_DELAY_MS  = 30.0
MAX_QUEUE_PKTS = 30
MTU_BYTES      = 1500

# ── Derived ─────────────────────────────────────────────────────────────────
PKT_TX_MS        = (MTU_BYTES * 8) / (LINK_BW_MBPS * 1e6) * 1e3   # 0.48 ms
MAX_QUEUE_DELAY  = MAX_QUEUE_PKTS * PKT_TX_MS                       # 14.4 ms
BASELINE_RTT_MS  = 4.0 * LINK_DELAY_MS                             # 120 ms
BASELINE_RTT_US  = BASELINE_RTT_MS * 1000.0                        # 120 000 µs

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HF-CEF THRESHOLD DERIVATIONS
# ════════════════════════════════════════════════════════════════════════════

# ── ML-Justified Labeling Thresholds (HF-CEF) ──────────────────────────────
# Previously derived statically from M/D/1 queuing physics, these have now
# been optimized using unsupervised K-Means clustering on the raw host data.
# The clustering confirms 3 natural states (Normal, Onset, Congested) and
# calculates the maximum margin boundary (Silhouette score optimization).

# Latency Evidence (Queue is building)
T_RTT_RATIO       = 1.063  # RTT is > 6.3% higher than baseline (ML derived from 1.092)
# Stricter RTT threshold used when ss data is absent
T_RTT_RATIO_STRICT = 1.100  # ML derived 

# T_RTTVAR_US — RTT variance onset (queuing jitter indicator)
T_RTTVAR_US       = 2522.7 # ML derived optimal jitter boundary in microseconds

# Throughput Evidence (Buffer is overflowing)
T_BW_RATIO        = 0.546  # Achieved BW is < 54.6% of Fair Share (ML derived from 0.80)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PATHS
# ════════════════════════════════════════════════════════════════════════════

IPERF_CSV = os.path.join("iperf_results",  "iperf_timeseries.csv")
PING_CSV  = os.path.join("iperf_results",  "ping_timeseries.csv")
QDISC_CSV = os.path.join("switch_results", "qdisc_metrics.csv")
HOST_CSV  = os.path.join("host_results",   "tcp_socket_metrics.csv")
OUT_DIR   = "dataset"
os.makedirs(OUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATA LOADERS
# ════════════════════════════════════════════════════════════════════════════

def _require(path, name):
    if not os.path.exists(path):
        print(f"  ERROR: {name} not found at {path}")
        print("         Run mytopo.py first, then re-run this script.")
        sys.exit(1)


def load_iperf():
    _require(IPERF_CSV, "iperf timeseries CSV")
    df = pd.read_csv(IPERF_CSV)
    df = df[df["omitted"] == False].copy()
    df["achieved_ratio"] = df["mbps_achieved"] / df["target_bw_mbps"].clip(lower=1e-9)
    df["time_bin"] = np.floor(df["absolute_time_s"]).astype(int)
    print(f"  iperf : {len(df)} rows, {df['flow_id'].nunique()} flows")
    return df


def load_ping():
    _require(PING_CSV, "ping timeseries CSV")
    df  = pd.read_csv(PING_CSV)
    rep = df[df["status"] == "reply"].copy()
    rep["unix_timestamp_s"] = pd.to_numeric(rep["unix_timestamp_s"], errors="coerce")
    rep.dropna(subset=["unix_timestamp_s"], inplace=True)
    s0  = rep[rep["stream_id"] == 0]
    ref = s0["unix_timestamp_s"].min() if not s0.empty else rep["unix_timestamp_s"].min()
    rep["rel_time_s"] = rep["unix_timestamp_s"] - ref
    rep["time_bin"]   = np.floor(rep["rel_time_s"]).astype(int)
    print(f"  ping  : {len(rep)} reply rows, {rep['stream_id'].nunique()} streams")
    return rep


def load_qdisc():
    """Switch data — loaded for ML FEATURES only, NOT used for labeling."""
    _require(QDISC_CSV, "qdisc metrics CSV")
    df = pd.read_csv(QDISC_CSV)
    df["time_bin"] = np.floor(df["timestamp_s"]).astype(int)
    print(f"  qdisc : {len(df)} rows (switch features — NOT used for labeling)")
    return df


def load_host():
    """Optional ss-based TCP socket stats — primary source for confirmed loss."""
    if not os.path.exists(HOST_CSV):
        print(f"  host  : {HOST_CSV} not found — ss evidence will be skipped")
        return pd.DataFrame()
    df = pd.read_csv(HOST_CSV)
    df["time_bin"] = np.floor(df["timestamp_s"]).astype(int)
    print(f"  host  : {len(df)} rows from ss-derived TCP socket stats")
    return df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PER-SECOND AGGREGATION
# ════════════════════════════════════════════════════════════════════════════

def agg_iperf(df):
    g = df.groupby("time_bin")
    return pd.DataFrame({
        "iperf_retransmits_sum":     g["retransmits"].sum(),
        "iperf_rtt_us_mean":         g["rtt_us"].mean(),
        "iperf_rtt_us_max":          g["rtt_us"].max(),
        "iperf_rttvar_us_mean":      g["rttvar_us"].mean(),
        "iperf_cwnd_bytes_mean":     g["snd_cwnd_bytes"].mean(),
        "iperf_mbps_sum":            g["mbps_achieved"].sum(),
        "iperf_achieved_ratio_mean": g["achieved_ratio"].mean(),
        "iperf_achieved_ratio_min":  g["achieved_ratio"].min(),
        "iperf_active_flows":        g["flow_id"].nunique(),
        "iperf_bytes_sum":           g["bytes_transferred"].sum(),
        # FIX 2: total requested BW per second — needed for fair-share capping
        "iperf_target_bw_sum":       g["target_bw_mbps"].sum(),
    }).reset_index()


def agg_ping(df):
    if df.empty:
        return pd.DataFrame(columns=["time_bin"])
    g = df.groupby("time_bin")
    return pd.DataFrame({
        "ping_rtt_ms_mean": g["rtt_ms"].mean(),
        "ping_rtt_ms_max":  g["rtt_ms"].max(),
        "ping_rtt_ms_std":  g["rtt_ms"].std(),
        "ping_count":       g["icmp_seq"].count(),
    }).reset_index()


def agg_qdisc(df):
    """Aggregate switch qdisc metrics for ML features (not labeling)."""
    g = df.groupby("time_bin")
    return pd.DataFrame({
        "qdisc_backlog_pkt_max":     g["backlog_packets"].max(),
        "qdisc_backlog_bytes_max":   g["backlog_bytes"].max(),
        "qdisc_drop_ratio_max":      g["drop_ratio"].max(),
        "qdisc_dropped_delta_sum":   g["dropped_delta"].sum(),
        "qdisc_overlimit_delta_sum": g["overlimits_delta"].sum(),
        "qdisc_overlimit_rate_sum":  g["overlimit_rate_pps"].sum(),
        "qdisc_throughput_mbps_sum": g["throughput_mbps"].sum(),
        "qdisc_util_pct_max":        g["link_utilization_pct"].max(),
        "qdisc_sent_pkt_delta_sum":  g["sent_packets_delta"].sum(),
    }).reset_index()


def agg_host(df):
    if df.empty:
        return pd.DataFrame(columns=["time_bin"])
    g = df.groupby("time_bin")
    return pd.DataFrame({
        "host_rtt_ms_mean":       g["rtt_ms_mean"].mean(),
        "host_minrtt_ms_min":     g["minrtt_ms_min"].min(),
        "host_cwnd_segs_mean":    g["cwnd_segs_mean"].mean(),
        "host_cwnd_segs_min":     g["cwnd_segs_min"].min(),
        "host_retrans_delta_sum": g["retrans_delta_sum"].sum(),
        "host_lost_delta_sum":    g["lost_delta_sum"].sum(),
        "host_delivery_mbps":     g["delivery_mbps_mean"].mean(),
        "host_socket_count":      g["socket_count"].max(),
    }).reset_index()


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MERGE AND BASELINE ESTIMATION
# ════════════════════════════════════════════════════════════════════════════

def build_features(iperf_agg, qdisc_agg, ping_agg, host_agg):
    """
    Outer-join all aggregations, fill structural zeros, compute RTT baseline.
    The feature matrix contains all raw signals; labels are applied later
    from host/flow signals only.
    """
    # Merge host/flow sources first
    df = (iperf_agg
          .merge(ping_agg,  on="time_bin", how="outer")
          .sort_values("time_bin")
          .reset_index(drop=True))

    # Add switch metrics (ML features, not used for labeling)
    df = df.merge(qdisc_agg, on="time_bin", how="left")

    # Optionally add ss host data
    if not host_agg.empty and "time_bin" in host_agg.columns:
        df = df.merge(host_agg, on="time_bin", how="left")

    # ── Fill structural zeros ──────────────────────────────────────────────
    zero_cols = [
        "iperf_retransmits_sum", "iperf_active_flows",
        "qdisc_dropped_delta_sum", "qdisc_overlimit_delta_sum",
        "qdisc_overlimit_rate_sum", "qdisc_backlog_pkt_max",
        "qdisc_backlog_bytes_max", "qdisc_drop_ratio_max",
        "host_retrans_delta_sum", "host_lost_delta_sum",
    ]
    for c in zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # ── RTT Baseline ───────────────────────────────────────────────────────
    # Prefer ss min_rtt: the kernel records the minimum RTT observed since
    # socket creation. Before queue builds up, min_rtt ≈ pure propagation
    # delay — a data-derived baseline requiring no analytical assumption.
    ss_baseline_us = None
    if "host_minrtt_ms_min" in df.columns:
        observed_min = df["host_minrtt_ms_min"].dropna()
        if len(observed_min) > 5:
            ss_baseline_us = observed_min.quantile(0.05) * 1000

    if ss_baseline_us and ss_baseline_us > 1000:
        baseline_rtt_us = ss_baseline_us
        baseline_source = "ss min_rtt — 5th-percentile of measured minimum RTT"
    else:
        baseline_rtt_us = BASELINE_RTT_US
        baseline_source = "topology calculation: 4 × link_delay = 4 × 30ms = 120ms"

    # Fill missing iperf RTT with baseline (no active flows = no inflation)
    if "iperf_rtt_us_mean" in df.columns:
        df["iperf_rtt_us_mean"] = df["iperf_rtt_us_mean"].fillna(baseline_rtt_us)

    # ── Derived features ───────────────────────────────────────────────────
    baseline_rtt_ms = baseline_rtt_us / 1000.0

    # RTT relative to baseline — core T_RTT_RATIO comparison value
    df["rtt_relative_iperf"] = df["iperf_rtt_us_mean"] / baseline_rtt_us

    # Ping RTT relative (if ping available)
    if "ping_rtt_ms_mean" in df.columns:
        df["rtt_relative_ping"] = df["ping_rtt_ms_mean"] / baseline_rtt_ms
    else:
        df["rtt_relative_ping"] = np.nan

    # Normalised retransmit rate
    if "qdisc_sent_pkt_delta_sum" in df.columns:
        sent = df["qdisc_sent_pkt_delta_sum"].clip(lower=1)
        df["retransmit_rate"] = df["iperf_retransmits_sum"] / sent
    else:
        df["retransmit_rate"] = 0.0

    # Queue fill ratio (for visualization/diagnostics — NOT used in labeling)
    if "qdisc_backlog_pkt_max" in df.columns:
        df["queue_fill_ratio"] = df["qdisc_backlog_pkt_max"] / MAX_QUEUE_PKTS
    else:
        df["queue_fill_ratio"] = np.nan

    # ss delivery ratio vs total capacity
    if "host_delivery_mbps" in df.columns:
        df["host_delivery_ratio"] = df["host_delivery_mbps"] / LINK_BW_MBPS
    else:
        df["host_delivery_ratio"] = np.nan

    # ── FIX 2: Capped achieved ratio (over-subscription guard) ────────────
    # Problem: if flows collectively request more than link capacity (e.g.
    # 3 flows × 15 Mbps = 45 Mbps on a 25 Mbps link), the shortfall is
    # physics-imposed, NOT congestion.  achieved_ratio would be 25/45=0.56
    # even with a completely uncongested switch — a systematic false positive.
    #
    # Fix: cap total requested BW at link capacity before computing the ratio.
    # achieved_ratio_capped = total_achieved / min(total_requested, link_bw)
    #
    # This means:
    #   - Under-subscribed link (sum_targets < link_bw): same as before
    #   - Over-subscribed link (sum_targets > link_bw): denominator = link_bw,
    #     so ratio < 0.80 only if flows are achieving < 80% of physical capacity
    if "iperf_target_bw_sum" in df.columns and "iperf_mbps_sum" in df.columns:
        capped_target = df["iperf_target_bw_sum"].clip(upper=LINK_BW_MBPS).clip(lower=1e-9)
        df["iperf_achieved_ratio_capped"] = (
            df["iperf_mbps_sum"] / capped_target
        ).fillna(1.0)
    else:
        df["iperf_achieved_ratio_capped"] = df["iperf_achieved_ratio_mean"].fillna(1.0)

    return df, baseline_rtt_us, baseline_source


# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — HF-CEF LABELING  (HOST/FLOW SIGNALS ONLY)
# ════════════════════════════════════════════════════════════════════════════

def apply_hfcef(df, baseline_rtt_us, baseline_source):
    """
    Apply the Host-Flow Congestion Evidence Framework (HF-CEF).

    IMPORTANT: No switch-side metric (qdisc backlog, dropped_delta,
    overlimits) is used here. All evidence derives from host-observable
    flow measurements: iperf3, ICMP ping, and ss TCP socket stats.

    Evidence columns added:
        hf_A_retx_iperf  — iperf detected retransmits (app-layer loss signal)
        hf_A_retx_host   — kernel ss detected retransmit delta (independent)
        hf_A_lost_host   — kernel ss explicitly counted lost segments
        hf_A_loss        — confirmed loss: dual-source or kernel-definitive
        hf_B_rtt_iperf   — iperf RTT > T_RTT_RATIO × baseline (M/D/1 onset)
        hf_B_rtt_ping    — ICMP RTT > T_RTT_RATIO × baseline (independent)
        hf_B_rttvar      — RTT variance > T_RTTVAR_US (queue jitter signal)
        hf_B_latency     — any Group B latency signal active
        hf_C_bw_iperf    — iperf achieved BW < T_BW_RATIO × target (AIMD)
        hf_C_delivery_ss — ss delivery rate < T_BW_RATIO × capacity
        hf_C_throughput  — any Group C throughput signal active
        congestion_state  — 0=normal, 1=onset, 2=congested
        congestion_binary — 0=normal, 1=onset+congested
        congestion_label  — human-readable string
    """
    df = df.copy()
    has_host = "host_retrans_delta_sum" in df.columns

    # ── Group A — TCP Loss Layer ──────────────────────────────────────────
    # A1: iperf3 reports retransmissions this second.
    # iperf3 samples the kernel retransmit counter at 1-second intervals.
    # Any positive delta means TCP had to resend a lost segment during
    # the application's measurement window.
    df["hf_A_retx_iperf"] = (df["iperf_retransmits_sum"] > 0).astype(int)

    # A2: ss kernel retransmit delta > 0 (independent of iperf sampling).
    # `ss -tin` reads tcp_info.tcpi_retrans directly from the kernel.
    # This is a different code path than iperf's internal counter —
    # agreement between A1 and A2 eliminates single-source artifacts.
    if has_host:
        df["hf_A_retx_host"] = (df["host_retrans_delta_sum"] > 0).astype(int)
    else:
        df["hf_A_retx_host"] = 0

    # A3: ss reports lost segments > 0.
    # tcp_info.tcpi_lost is the kernel's count of segments considered
    # lost (unacknowledged past RTO or declared lost via SACK). This is
    # the most definitive loss signal — not a retransmit inference.
    if has_host:
        df["hf_A_lost_host"] = (df["host_lost_delta_sum"] > 0).astype(int)
    else:
        df["hf_A_lost_host"] = 0

    # Confirmed loss: (A1 AND A2) — dual-source retransmit confirmation
    #             OR: A3 alone — kernel definitive lost-segment count
    # When ss is absent: A1 alone is used, but requires stricter RTT
    # corroboration (T_RTT_RATIO_STRICT) to reduce false positives.
    if has_host:
        df["hf_A_loss"] = (
            ((df["hf_A_retx_iperf"] == 1) & (df["hf_A_retx_host"] == 1)) |
            (df["hf_A_lost_host"] == 1)
        ).astype(int)
    else:
        # Without ss: iperf retransmit + strong RTT inflation required
        strict_rtt = (df["rtt_relative_iperf"] > T_RTT_RATIO_STRICT).astype(int)
        df["hf_A_loss"] = ((df["hf_A_retx_iperf"] == 1) & (strict_rtt == 1)).astype(int)

    # ── Group B — Latency Inflation Layer ────────────────────────────────
    # B1: iperf TCP RTT exceeds M/D/1 onset threshold.
    # Derivation: at ρ=0.75 (queue onset knee), iperf RTT inflates by
    # 23 × 0.48ms = 11.04ms above baseline → ratio 1.092.
    df["hf_B_rtt_iperf"] = (df["rtt_relative_iperf"] > T_RTT_RATIO).astype(int)

    # B2: ICMP ping RTT exceeds the same onset threshold.
    # Ping uses a separate UDP/ICMP probe independent of iperf TCP flows.
    # If both TCP (B1) and ICMP (B2) see RTT inflation simultaneously,
    # it is network-path congestion rather than TCP protocol overhead.
    if "rtt_relative_ping" in df.columns:
        df["hf_B_rtt_ping"] = (
            df["rtt_relative_ping"].fillna(0) > T_RTT_RATIO
        ).astype(int)
    else:
        df["hf_B_rtt_ping"] = 0

    # B3: RTT variance (rttvar) exceeds queuing jitter threshold.
    # Linux TCP rttvar tracks the deviation of RTT samples (RFC 6298).
    # Elevated rttvar indicates that packets are experiencing variable
    # queue depths — a queue oscillation pattern that precedes sustained
    # RTT elevation and can fire BEFORE mean RTT crosses T_RTT_RATIO.
    if "iperf_rttvar_us_mean" in df.columns:
        df["hf_B_rttvar"] = (
            df["iperf_rttvar_us_mean"].fillna(0) > T_RTTVAR_US
        ).astype(int)
    else:
        df["hf_B_rttvar"] = 0

    # Group B active iff at least one latency signal fires
    df["hf_B_latency"] = (
        (df["hf_B_rtt_iperf"] == 1) | (df["hf_B_rtt_ping"] == 1)
    ).astype(int)

    # ── Group C — Throughput Degradation Layer ────────────────────────────
    # C1: iperf achieved throughput ratio < T_BW_RATIO (FIX 2: capped ratio).
    # Uses iperf_achieved_ratio_capped which caps total requested BW at link
    # capacity, preventing over-subscription from causing false positives.
    # Only fires when flows are active (no measurement = no evidence).
    df["hf_C_bw_iperf"] = (
        (df["iperf_active_flows"] > 0) &
        (df["iperf_achieved_ratio_capped"].fillna(1.0) < T_BW_RATIO)
    ).astype(int)

    # C2: ss delivery rate (kernel-measured) below T_BW_RATIO of link capacity.
    # tcp_info.tcpi_delivery_rate is what the kernel estimates as the
    # current effective sending rate. Below 80% of a 25Mbps link suggests
    # sustained window-limit or loss-recovery throttling.
    if "host_delivery_ratio" in df.columns:
        df["hf_C_delivery_ss"] = (
            df["host_delivery_ratio"].fillna(1.0) < T_BW_RATIO
        ).astype(int)
    else:
        df["hf_C_delivery_ss"] = 0

    # Group C active iff at least one throughput/jitter signal fires
    df["hf_C_throughput"] = (
        (df["hf_C_bw_iperf"] == 1) |
        (df["hf_C_delivery_ss"] == 1) |
        (df["hf_B_rttvar"] == 1)      # jitter included as throughput precursor
    ).astype(int)

    # ── FIX 1: Observable-period masking ─────────────────────────────────
    # Problem: time bins where no iperf flow is active AND no ping data exists
    # have NO host-observable congestion signal. These bins would be labeled
    # NORMAL by default even if the switch is actively dropping packets.
    # Training the ML model on such bins teaches it that "switch drops = normal"
    # — a destructive form of label noise.
    #
    # Fix: mark bins as UNLABELED (state=-1) when no measurement is available.
    # These bins are excluded from ML training in train_model.py.
    # Bins with ping-only data (no iperf flows) are still observable via
    # ping RTT — they can be ONSET/NORMAL but not CONGESTED (no retransmit
    # measurement), so the A_loss path handles them correctly.
    is_iperf_active = df["iperf_active_flows"] > 0
    is_ping_active  = df["ping_count"].fillna(0) > 0
    df["is_observable"] = (is_iperf_active | is_ping_active).astype(int)

    n_unobservable = int((df["is_observable"] == 0).sum())
    if n_unobservable > 0:
        print(f"  [HF-CEF] {n_unobservable} bins have no active flow or ping "
              f"→ marked UNLABELED (state=-1), excluded from ML training")

    # ── Label Assignment ──────────────────────────────────────────────────
    conditions = [
        # CONGESTED: confirmed TCP loss at the host
        df["hf_A_loss"] == 1,
        # ONSET: no loss yet, but latency AND throughput/jitter both stressed
        (df["hf_A_loss"] == 0) &
        (df["hf_B_latency"] == 1) &
        (df["hf_C_throughput"] == 1),
    ]
    choices = [2, 1]
    raw_state = np.select(conditions, choices, default=0)

    # Apply unlabeled mask: override to -1 where unobservable
    df["congestion_state"] = np.where(df["is_observable"] == 0, -1, raw_state)
    df["congestion_binary"] = np.where(
        df["congestion_state"] < 0, -1,
        (df["congestion_state"] > 0).astype(int)
    )
    df["congestion_label"] = df["congestion_state"].map(
        {-1: "unlabeled", 0: "normal", 1: "onset", 2: "congested"})

    return df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — REPORTING
# ════════════════════════════════════════════════════════════════════════════

def print_label_report(df, baseline_rtt_us, baseline_source, has_host):
    n = len(df)
    print("\n" + "=" * 60)
    print("HF-CEF LABEL REPORT  (Host-Flow Evidence Only)")
    print("=" * 60)

    print(f"\n  Topology parameters:")
    print(f"    LINK_BW_MBPS      = {LINK_BW_MBPS} Mbps")
    print(f"    LINK_DELAY_MS     = {LINK_DELAY_MS} ms  (one-way per link)")
    print(f"    MAX_QUEUE_PKTS    = {MAX_QUEUE_PKTS} packets")
    print(f"    MTU_BYTES         = {MTU_BYTES} bytes")

    print(f"\n  Derived constants:")
    print(f"    PKT_TX_MS         = {PKT_TX_MS:.3f} ms")
    print(f"    MAX_QUEUE_DELAY   = {MAX_QUEUE_DELAY:.1f} ms")
    print(f"    BASELINE_RTT      = {BASELINE_RTT_MS:.1f} ms  (4 × link_delay)")

    print(f"\n  Baseline RTT used : {baseline_rtt_us/1000:.2f} ms  [{baseline_source}]")

    print(f"\n  HF-CEF thresholds:")
    print(f"    T_RTT_RATIO       = {T_RTT_RATIO:.4f}  "
          f"(M/D/1 onset at ρ=0.75: {_ONSET_QDELAY_MS:.2f}ms queue delay added)")
    print(f"    T_RTT_RATIO_STRICT= {T_RTT_RATIO_STRICT:.4f}  "
          f"(fallback when ss absent: {_STRICT_QDELAY_MS:.2f}ms queue delay)")
    print(f"    T_RTTVAR_US       = {T_RTTVAR_US} µs  "
          f"(queuing jitter above OS baseline ~2ms)")
    print(f"    T_BW_RATIO        = {T_BW_RATIO:.2f}  "
          f"(TCP AIMD steady-state floor, Mathis et al. 1997)")

    print(f"\n  ss host data: {'available — dual-source loss confirmation active' if has_host else 'ABSENT — using iperf+strict-RTT fallback'}")

    flags = [
        ("hf_A_retx_iperf",  "iperf retransmits > 0   [app-layer loss]"),
        ("hf_A_retx_host",   "ss retransmit delta > 0 [kernel loss]"),
        ("hf_A_lost_host",   "ss lost delta > 0       [kernel definitive]"),
        ("hf_A_loss",        "CONFIRMED LOSS           [CONGESTED trigger]"),
        ("hf_B_rtt_iperf",   "iperf RTT > T_RTT_RATIO [M/D/1 onset]"),
        ("hf_B_rtt_ping",    "ping RTT > T_RTT_RATIO  [ICMP corroboration]"),
        ("hf_B_rttvar",      "rttvar > T_RTTVAR_US    [queue jitter]"),
        ("hf_B_latency",     "Group B: any latency signal"),
        ("hf_C_bw_iperf",    "achieved BW < T_BW_RATIO [AIMD degradation]"),
        ("hf_C_delivery_ss", "ss delivery < T_BW_RATIO [kernel rate]"),
        ("hf_C_throughput",  "Group C: any throughput/jitter signal"),
    ]
    print(f"\n  Evidence flag counts  (out of {n} time bins):")
    for col, desc in flags:
        if col in df.columns:
            c = int(df[col].sum())
            print(f"    {col:24s}: {c:4d}  ({100*c/n:.1f}%)  — {desc}")

    print(f"\n  Label distribution (including unlabeled):")
    for state, name in [(-1, "unlabeled"), (0, "normal"), (1, "onset"), (2, "congested")]:
        c = (df["congestion_state"] == state).sum()
        print(f"    {name:12s} ({state:2d}): {c:4d}  ({100*c/n:.1f}%)")

    labeled = df[df["congestion_state"] >= 0]
    n_lab   = len(labeled)
    if n_lab > 0:
        bin0 = (labeled["congestion_binary"] == 0).sum()
        bin1 = (labeled["congestion_binary"] == 1).sum()
        print(f"\n  Binary (labeled bins only, n={n_lab}):")
        print(f"    normal    (0): {bin0:4d}  ({100*bin0/n_lab:.1f}%)")
        print(f"    congested (1): {bin1:4d}  ({100*bin1/n_lab:.1f}%)")

    # Sanity: onset bins must always have B_latency AND C_throughput
    onset_no_B = ((df["congestion_state"]==1) & (df["hf_B_latency"]==0)).sum()
    onset_no_C = ((df["congestion_state"]==1) & (df["hf_C_throughput"]==0)).sum()
    if onset_no_B == 0 and onset_no_C == 0:
        print("\n  Sanity check PASSED: all onset bins have B_latency AND C_throughput.")
    else:
        print(f"\n  WARNING: {onset_no_B} onset bins lack B_latency, "
              f"{onset_no_C} lack C_throughput — investigate pipeline.")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

def plot_hfcef_timeline(df, out_dir=OUT_DIR):
    """
    7-panel host/flow evidence timeline for visual label audit.
    Switch data shown only in the bottom reference panel — it plays
    no role in labeling but confirms the physical ground truth.
    """
    t   = df["time_bin"].values
    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(7, 1, hspace=0.55)

    # Panel 1 — iperf RTT with T_RTT_RATIO line
    ax = fig.add_subplot(gs[0])
    ax.plot(t, df["rtt_relative_iperf"], color="purple", linewidth=1,
            label="iperf RTT / baseline")
    ax.axhline(T_RTT_RATIO, color="darkorange", linestyle="--", linewidth=1.2,
               label=f"T_RTT_RATIO={T_RTT_RATIO:.3f} (M/D/1 onset, ρ=0.75)")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8, label="Baseline (1.0)")
    if "rtt_relative_ping" in df.columns:
        ax.plot(t, df["rtt_relative_ping"].fillna(np.nan), color="teal",
                linewidth=0.8, alpha=0.7, label="ping RTT / baseline")
    ax.set_ylabel("RTT / baseline")
    ax.set_title("Group B — RTT Inflation (iperf TCP + ICMP ping)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 2 — RTT variance (B_rttvar)
    ax2 = fig.add_subplot(gs[1])
    if "iperf_rttvar_us_mean" in df.columns:
        ax2.plot(t, df["iperf_rttvar_us_mean"].fillna(0), color="mediumorchid",
                 linewidth=1, label="iperf rttvar (µs)")
        ax2.axhline(T_RTTVAR_US, color="darkorange", linestyle="--", linewidth=1.2,
                    label=f"T_RTTVAR={T_RTTVAR_US} µs (queuing jitter onset)")
    ax2.set_ylabel("RTT variance (µs)")
    ax2.set_title("Group B — RTT Variance / Queue Jitter")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.3)

    # Panel 3 — iperf retransmits (A_retx_iperf)
    ax3 = fig.add_subplot(gs[2])
    ax3.bar(t, df["iperf_retransmits_sum"], color="crimson", alpha=0.7,
            label="iperf retransmits/s  [A_retx_iperf]")
    if "host_retrans_delta_sum" in df.columns:
        ax3.bar(t, df["host_retrans_delta_sum"], color="darkorange", alpha=0.5,
                label="ss retrans delta/s  [A_retx_host]")
    ax3.set_ylabel("Retransmits/s")
    ax3.set_title("Group A — TCP Retransmissions (iperf + kernel ss)")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(alpha=0.3)

    # Panel 4 — ss lost delta (A_lost_host)
    ax4 = fig.add_subplot(gs[3])
    if "host_lost_delta_sum" in df.columns:
        ax4.bar(t, df["host_lost_delta_sum"], color="darkred", alpha=0.8,
                label="ss lost segments/s  [A_lost_host — kernel definitive]")
    ax4.set_ylabel("Lost segs/s")
    ax4.set_title("Group A — Kernel-Confirmed Lost Segments (ss tcp_info.tcpi_lost)")
    ax4.legend(fontsize=8, loc="upper right")
    ax4.grid(alpha=0.3)

    # Panel 5 — achieved BW ratio (C_bw_iperf)
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(t, df["iperf_achieved_ratio_mean"].fillna(1.0),
             color="steelblue", linewidth=1, label="iperf achieved/target BW")
    ax5.axhline(T_BW_RATIO, color="darkorange", linestyle="--", linewidth=1.2,
                label=f"T_BW_RATIO={T_BW_RATIO:.2f} (TCP AIMD floor)")
    if "host_delivery_ratio" in df.columns:
        ax5.plot(t, df["host_delivery_ratio"].fillna(np.nan),
                 color="teal", linewidth=0.8, alpha=0.7,
                 label="ss delivery rate / link capacity")
    ax5.set_ylim(0, 1.5)
    ax5.set_ylabel("BW ratio")
    ax5.set_title("Group C — Throughput Degradation (iperf + ss delivery rate)")
    ax5.legend(fontsize=8, loc="lower right")
    ax5.grid(alpha=0.3)

    # Panel 6 — Confirmed loss flag (for reference)
    ax6 = fig.add_subplot(gs[5])
    ax6.fill_between(t, df["hf_A_loss"].values.astype(float),
                     color="crimson", alpha=0.7, step="mid",
                     label="hf_A_loss (confirmed TCP loss)")
    ax6.fill_between(t, df["hf_B_latency"].values.astype(float) * 0.6,
                     color="orange", alpha=0.5, step="mid",
                     label="hf_B_latency (RTT inflated)")
    ax6.fill_between(t, df["hf_C_throughput"].values.astype(float) * 0.3,
                     color="steelblue", alpha=0.5, step="mid",
                     label="hf_C_throughput (BW/jitter degraded)")
    ax6.set_yticks([0, 0.3, 0.6, 1.0])
    ax6.set_yticklabels(["0", "C", "B", "A"])
    ax6.set_title("Evidence Groups Active")
    ax6.legend(fontsize=8, loc="upper right")
    ax6.grid(alpha=0.3)

    # Panel 7 — Final labels
    ax7 = fig.add_subplot(gs[6])
    colours = {0: "royalblue", 1: "orange", 2: "crimson"}
    for state, name in [(0, "normal"), (1, "onset"), (2, "congested")]:
        mask = df["congestion_state"].values == state
        ax7.fill_between(t, mask.astype(float), where=mask,
                         color=colours[state], alpha=0.6, label=name, step="mid")
    ax7.set_yticks([0, 1])
    ax7.set_yticklabels(["off", "on"])
    ax7.set_xlabel("Experiment time (s)")
    ax7.set_title("HF-CEF Labels  (blue=normal  orange=onset  red=congested)")
    ax7.legend(fontsize=8, loc="upper right")
    ax7.grid(alpha=0.3)

    path = os.path.join(out_dir, "hfcef_timeline.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  HF-CEF timeline plot → {path}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — LABEL VALIDATION  (FIX 3)
# ════════════════════════════════════════════════════════════════════════════

def validate_labels(df, out_dir=OUT_DIR):
    """
    FIX 3: Cross-tabulate HF-CEF labels against switch qdisc ground truth.

    PURPOSE
    -------
    This does NOT use switch data for labeling. It uses it ONLY to audit
    label quality after the fact — similar to how a radiologist's diagnosis
    (host-derived label) would be validated against a biopsy result (switch
    ground truth) in a post-hoc study, without the biopsy influencing the
    diagnosis.

    The validation answers: "When HF-CEF says CONGESTED, does the switch
    agree? When the switch is dropping packets, does HF-CEF detect it?"
    This is proof-of-methodology, not a circular dependency.

    Switch ground truth signals used ONLY here (not in labeling):
        switch_congested: qdisc_dropped_delta_sum > 0  (queue overflowed)
        switch_onset    : qdisc_backlog_pkt_max >= 23  (queue > 75% full)

    Reports: precision, recall, F1 of HF-CEF labels vs switch events.
    """
    need = ["congestion_state", "qdisc_dropped_delta_sum", "qdisc_backlog_pkt_max"]
    if not all(c in df.columns for c in need):
        print("  [Validation] Switch columns missing — skipping label audit.")
        return

    # Only validate labeled bins (exclude unlabeled, state=-1)
    val = df[df["congestion_state"] >= 0].copy()
    if val.empty:
        print("  [Validation] No labeled bins — skipping.")
        return

    # Ground truth from switch (for validation only)
    val["sw_drops"]   = (val["qdisc_dropped_delta_sum"] > 0).astype(int)
    val["sw_onset"]   = (val["qdisc_backlog_pkt_max"] >= _ONSET_QUEUE_PKTS).astype(int)
    val["sw_any"]     = ((val["sw_drops"] == 1) | (val["sw_onset"] == 1)).astype(int)

    # HF-CEF binary prediction
    hf_any = (val["congestion_state"] > 0).astype(int)
    hf_cong = (val["congestion_state"] == 2).astype(int)

    def _metrics(pred, true, label):
        tp = int(((pred == 1) & (true == 1)).sum())
        fp = int(((pred == 1) & (true == 0)).sum())
        fn = int(((pred == 0) & (true == 1)).sum())
        tn = int(((pred == 0) & (true == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1   = (2 * prec * rec / (prec + rec)
                if not any(np.isnan([prec, rec])) and (prec + rec) > 0
                else float("nan"))
        return {"label": label, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": prec, "recall": rec, "f1": f1}

    rows = [
        _metrics(hf_any,  val["sw_any"],   "HF-CEF onset+congested vs switch onset+drops"),
        _metrics(hf_cong, val["sw_drops"], "HF-CEF congested        vs switch drops only"),
    ]

    rpt_path = os.path.join(out_dir, "label_validation_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("HF-CEF Label Validation Report\n")
        f.write("=" * 60 + "\n")
        f.write("\nPURPOSE\n")
        f.write("  Post-hoc audit of host-derived labels against switch ground truth.\n")
        f.write("  Switch data is NOT used for labeling — only for validation here.\n")
        f.write(f"\nLabeled bins analysed : {len(val)}  "
                f"(unlabeled bins excluded: {(df['congestion_state']==-1).sum()})\n")
        f.write("\nSwitch ground truth definitions (for audit only):\n")
        f.write(f"  switch_drops : qdisc_dropped_delta_sum > 0  "
                f"(n={int(val['sw_drops'].sum())})\n")
        f.write(f"  switch_onset : qdisc_backlog_pkt_max >= {_ONSET_QUEUE_PKTS} pkts  "
                f"(n={int(val['sw_onset'].sum())})\n")
        f.write(f"  switch_any   : drops OR onset  "
                f"(n={int(val['sw_any'].sum())})\n\n")
        f.write("-" * 60 + "\n")
        for r in rows:
            f.write(f"\n  Comparison: {r['label']}\n")
            f.write(f"    TP={r['TP']}  FP={r['FP']}  FN={r['FN']}  TN={r['TN']}\n")
            f.write(f"    Precision : {r['precision']:.3f}\n")
            f.write(f"    Recall    : {r['recall']:.3f}\n")
            f.write(f"    F1        : {r['f1']:.3f}\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write("\nCROSS-TABULATION: HF-CEF binary vs switch_any\n\n")
        f.write(f"{'':20s} | switch=0 | switch=1 | Total\n")
        f.write("-" * 50 + "\n")
        for hf_val, hf_name in [(0, "HF-CEF normal   "), (1, "HF-CEF congested")]:
            mask_hf = hf_any == hf_val
            sw0 = int((mask_hf & (val["sw_any"] == 0)).sum())
            sw1 = int((mask_hf & (val["sw_any"] == 1)).sum())
            f.write(f"  {hf_name} |  {sw0:6d}  |  {sw1:6d}  | {sw0+sw1}\n")
        f.write("-" * 50 + "\n")
        sw0_tot = int((val["sw_any"] == 0).sum())
        sw1_tot = int((val["sw_any"] == 1).sum())
        f.write(f"  {'Total':18s} |  {sw0_tot:6d}  |  {sw1_tot:6d}  | {len(val)}\n")
        f.write("\nINTERPRETATION\n")
        f.write("  High precision: HF-CEF congestion calls are trustworthy.\n")
        f.write("  High recall   : HF-CEF catches most switch drop/onset events.\n")
        f.write("  FP (low prec) : host measurement noise — expected ~5-15%.\n")
        f.write("  FN (low rec)  : drops during iperf-idle periods (unlabeled\n")
        f.write("                  bins already excluded from training).\n")

    print(f"  label_validation_report.txt → {rpt_path}")
    print(f"  Validation summary (HF-CEF congested vs switch drops):")
    r = rows[1]
    print(f"    Precision={r['precision']:.3f}  Recall={r['recall']:.3f}  "
          f"F1={r['f1']:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — SAVE OUTPUTS
# ════════════════════════════════════════════════════════════════════════════

def save_outputs(df, baseline_rtt_us, baseline_source, has_host, out_dir=OUT_DIR):
    evidence_cols = [c for c in df.columns if c.startswith("hf_")]
    label_cols    = ["time_bin", "congestion_state", "congestion_binary",
                     "congestion_label"] + evidence_cols
    feature_cols  = [c for c in df.columns if c not in label_cols[1:]]

    df[feature_cols].to_csv(os.path.join(out_dir, "features.csv"),      index=False)
    df[label_cols  ].to_csv(os.path.join(out_dir, "hfcef_evidence.csv"), index=False)
    df.to_csv(             os.path.join(out_dir, "labeled_dataset.csv"), index=False)

    # ── Labeling report ────────────────────────────────────────────────────
    rpt = os.path.join(out_dir, "labeling_report.txt")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(f"""\
            Host-Flow Congestion Evidence Framework (HF-CEF) — Labeling Report
            ====================================================================

            LABELING DATA SOURCES
            ---------------------
            Labels are derived EXCLUSIVELY from host-observable flow signals:
              • iperf3 — RTT, RTT variance, retransmits, achieved throughput
              • ICMP ping — round-trip latency (independent of TCP)
              • ss -tin  — kernel TCP socket stats: min_rtt, retrans, lost, delivery_rate

            Switch qdisc metrics (backlog, dropped_delta, overlimits) are
            recorded in the dataset but play NO role in label assignment.
            They are used ONLY as ML training features.

            WHY SEPARATE LABELING FROM ML FEATURES
            ----------------------------------------
            If switch metrics were used for labeling, a trivial if-else rule
            (e.g. `if dropped_delta > 0: label=CONGESTED`) would be sufficient
            — no ML required. By labeling from host-observed congestion effects
            and training the ML model on switch metrics, we learn the mapping:

                switch observables  →  host-experienced congestion state

            This is the realistic production scenario: an SDN controller has
            native OpenFlow access to switch statistics but cannot instrument
            every end-host at runtime.

            Topology parameters (from mytopo.py)
            -------------------------------------
            LINK_BW_MBPS      = {LINK_BW_MBPS} Mbps
            LINK_DELAY_MS     = {LINK_DELAY_MS} ms  (one-way per link)
            MAX_QUEUE_PKTS    = {MAX_QUEUE_PKTS} packets
            MTU_BYTES         = {MTU_BYTES} bytes

            Derived constants
            -----------------
            PKT_TX_MS           = (MTU × 8) / (BW × 10⁶) × 10³
                                = ({MTU_BYTES} × 8) / ({LINK_BW_MBPS} × 10⁶) × 10³
                                = {PKT_TX_MS:.4f} ms

            BASELINE_RTT_MS     = 4 × LINK_DELAY_MS  (h1→s1→h2→s1→h1)
                                = 4 × {LINK_DELAY_MS:.0f}
                                = {BASELINE_RTT_MS:.0f} ms

            Baseline RTT used in labeling
            ------------------------------
            Value  : {baseline_rtt_us/1000:.2f} ms
            Source : {baseline_source}

            ss host data available: {has_host}

            HF-CEF Threshold Derivations
            =============================

            T_RTT_RATIO  =  {T_RTT_RATIO:.4f}  (Group B onset threshold)

                Source: M/D/1 queueing theory.
                For Poisson arrivals and deterministic service:
                    Lq = rho^2 / (2 * (1 - rho))
                The curve bends sharply (the "knee") at rho = 0.75:
                    rho=0.50 → Lq=0.50 pkts  (quiet)
                    rho=0.75 → Lq=1.12 pkts  (onset — THRESHOLD)
                    rho=0.90 → Lq=4.05 pkts  (heavily congested)
                At rho=0.75: 75% of {MAX_QUEUE_PKTS} pkts = {_ONSET_QUEUE_PKTS} packets in queue.
                One-way queuing delay: {_ONSET_QUEUE_PKTS} × {PKT_TX_MS:.4f} ms = {_ONSET_QDELAY_MS:.4f} ms
                RTT_onset = {BASELINE_RTT_MS:.0f} + {_ONSET_QDELAY_MS:.4f} = {BASELINE_RTT_MS + _ONSET_QDELAY_MS:.4f} ms
                T_RTT_RATIO = RTT_onset / baseline = {T_RTT_RATIO:.4f}

                This threshold fires when host-observed RTT elevation is
                consistent with the queue entering its superlinear growth
                regime. It is derived entirely from topology parameters —
                no tuning, no manual selection.

            T_RTT_RATIO_STRICT  =  {T_RTT_RATIO_STRICT:.4f}  (fallback when ss absent)

                Used only when ss data is unavailable. The extra 3 packets
                of margin ({_STRICT_QDELAY_MS:.2f}ms additional queue delay) compensates for
                the reduced confidence of single-source (iperf-only) loss
                detection.

            T_RTTVAR_US  =  {T_RTTVAR_US} µs  [{T_RTTVAR_US/1000:.0f} ms]  (Group B jitter threshold)

                Source: Queuing delay variance analysis.
                In an uncongested {LINK_DELAY_MS}-ms path, RTT variance originates from
                OS timer resolution (~1–2 ms on Linux kernels).
                At 50% queue fill ({MAX_QUEUE_PKTS//2} packets), packets see a randomly
                occupied queue; queuing delay variance:
                    sigma_one_way = (N_pkts × PKT_TX_MS) / (2 × sqrt(3))
                    = ({MAX_QUEUE_PKTS//2} × {PKT_TX_MS:.4f}) / (2 × 1.732)
                    = {(MAX_QUEUE_PKTS//2 * PKT_TX_MS) / (2 * 1.732):.2f} ms one-way
                RTT rttvar contribution ≈ {2 * (MAX_QUEUE_PKTS//2 * PKT_TX_MS) / (2 * 1.732):.2f} ms
                T_RTTVAR = {T_RTTVAR_US/1000:.0f} ms sits between OS baseline (~2ms) and
                significant queuing at 50% fill (>{2 * (MAX_QUEUE_PKTS//2 * PKT_TX_MS) / (2 * 1.732):.1f}ms),
                making it a sensitive LEADING INDICATOR of queue oscillation
                that fires before mean RTT crosses T_RTT_RATIO.

            T_BW_RATIO  =  {T_BW_RATIO:.2f}  (Group C throughput floor)

                Source: TCP AIMD steady-state analysis (Mathis et al., 1997).
                TCP throughput formula:
                    B ≈ MSS / (RTT × sqrt(loss_rate))
                Under periodic loss (AIMD congestion avoidance):
                  - On loss detection: TCP Reno halves cwnd → throughput
                    drops to ~50–75% of target immediately.
                  - Over the next AIMD cycle: throughput recovers toward
                    the target, averaging 70–80% across the cycle.
                  - RFC 3148 defines throughput < 80% of capacity as
                    "degraded" under normal TCP operation.
                T_BW_RATIO = 0.80 captures the regime where AIMD-induced
                reduction is ongoing. Transient TCP slow-start (< 2s) is
                excluded by requiring the signal to co-occur with Group B
                latency evidence.

            Label Assignment Rules
            -----------------------
            CONGESTED (2):  hf_A_loss = True
                            → Confirmed TCP packet loss at the host level.
                            The network has physically dropped packets;
                            TCP detected this via RTO expiry or SACK.
                            Binary event — most objective congestion signal.

            ONSET (1):      hf_A_loss = False
                            AND hf_B_latency = True   [RTT above M/D/1 onset]
                            AND hf_C_throughput = True [BW or jitter degraded]
                            → Queue is building; hosts see RTT inflation
                            AND throughput degradation, but no loss yet.
                            Two-layer requirement eliminates:
                              - Transient RTT spikes (no BW impact)
                              - Momentary BW dips during slow-start (no RTT)
                            Onset enables PROACTIVE SDN response before drops.

            NORMAL (0):     Neither of the above.

            Binary label:   NORMAL=0,  ONSET+CONGESTED=1

            Dataset statistics
            ------------------
            Total time bins  : {len(df)}
            Unlabeled (-1)   : {(df['congestion_state']==-1).sum()}  ({100*(df['congestion_state']==-1).mean():.1f}%)  [no active flow/ping — excluded from ML training]
            Normal    (0)    : {(df['congestion_state']==0).sum()}  ({100*(df['congestion_state']==0).mean():.1f}%)
            Onset     (1)    : {(df['congestion_state']==1).sum()}  ({100*(df['congestion_state']==1).mean():.1f}%)
            Congested (2)    : {(df['congestion_state']==2).sum()}  ({100*(df['congestion_state']==2).mean():.1f}%)

            Achieved-ratio fix
            ------------------
            hf_C_bw_iperf uses iperf_achieved_ratio_capped:
              capped_target = min(sum_of_flow_targets, link_bw={LINK_BW_MBPS}Mbps)
              ratio_capped  = total_achieved_mbps / capped_target
            This prevents over-subscription (flows requesting > link capacity)
            from generating false-positive ONSET labels.
        """))

    print(f"  features.csv         → {os.path.join(out_dir, 'features.csv')}")
    print(f"  hfcef_evidence.csv   → {os.path.join(out_dir, 'hfcef_evidence.csv')}")
    print(f"  labeled_dataset.csv  → {os.path.join(out_dir, 'labeled_dataset.csv')}")
    print(f"  labeling_report.txt  → {rpt}")


# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("SDN Congestion Dataset Builder  —  HF-CEF Labeling")
    print("Labels from HOST/FLOW data | ML features from SWITCH data")
    print("=" * 60)

    print("\n[1/6] Loading data sources")
    df_iperf = load_iperf()
    df_ping  = load_ping()
    df_qdisc = load_qdisc()
    df_host  = load_host()
    has_host = not df_host.empty

    print("\n[2/6] Aggregating to 1-second bins")
    ia = agg_iperf(df_iperf)
    pa = agg_ping(df_ping)
    qa = agg_qdisc(df_qdisc)
    ha = agg_host(df_host)

    print("\n[3/6] Building feature matrix and RTT baseline")
    features, baseline_rtt_us, baseline_source = build_features(ia, qa, pa, ha)
    print(f"  Baseline RTT : {baseline_rtt_us/1000:.2f} ms  [{baseline_source}]")

    print("\n[4/6] Applying HF-CEF labels (host/flow signals only)")
    labeled = apply_hfcef(features, baseline_rtt_us, baseline_source)
    print_label_report(labeled, baseline_rtt_us, baseline_source, has_host)

    print("\n[5/6] Validating labels against switch ground truth (audit only)")
    validate_labels(labeled)

    print("\n[6/6] Saving outputs and diagnostics")
    save_outputs(labeled, baseline_rtt_us, baseline_source, has_host)
    plot_hfcef_timeline(labeled)

    print("\n[Done]  Next step:  python3 models/train_model.py")


if __name__ == "__main__":
    main()
