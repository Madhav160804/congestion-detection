#!/usr/bin/env python3
"""
Synthetic Dataset Generator — SDN Congestion Detection
=======================================================
Generates realistic CSV files matching the Mininet experiment format.
Uses M/D/1 queuing theory + TCP AIMD model to simulate congestion.

Topology simulated:
    h1 --[30ms, 25Mbps, queue=30]--> s1 --> h2

Outputs:
    iperf_results/iperf_timeseries.csv
    iperf_results/ping_timeseries.csv
    switch_results/qdisc_metrics.csv
    host_results/tcp_socket_metrics.csv
"""

import os
import numpy as np
import pandas as pd

# ── Topology constants (must match build_dataset.py) ──────────────────────
LINK_BW_MBPS    = 25.0
LINK_DELAY_MS   = 30.0
MAX_QUEUE_PKTS  = 30
MTU_BYTES       = 1500
PKT_TX_MS       = (MTU_BYTES * 8) / (LINK_BW_MBPS * 1e6) * 1e3   # 0.48 ms
BASELINE_RTT_MS = 4.0 * LINK_DELAY_MS                             # 120 ms
BASELINE_RTT_US = BASELINE_RTT_MS * 1000.0

DURATION_S  = 10000
UNIX_T0     = 1_700_000_000.0
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

IPERF_DIR  = "iperf_results"
SWITCH_DIR = "switch_results"
HOST_DIR   = "host_results"
for d in [IPERF_DIR, SWITCH_DIR, HOST_DIR]:
    os.makedirs(d, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# PHYSICAL MODEL
# ════════════════════════════════════════════════════════════════════════════

def md1_lq(rho: float) -> float:
    """M/D/1 mean queue occupancy: Lq = rho^2 / (2*(1-rho))"""
    rho = float(np.clip(rho, 0.0, 0.9995))
    return rho ** 2 / (2.0 * (1.0 - rho))


def rtt_from_queue(queue_pkts: float) -> float:
    """RTT in ms given queue depth in packets."""
    return BASELINE_RTT_MS + queue_pkts * PKT_TX_MS


# ════════════════════════════════════════════════════════════════════════════
# FLOW SCHEDULE
# ════════════════════════════════════════════════════════════════════════════

def generate_flow_schedule() -> list:
    """
    Scale-up scheduling for thousands of seconds.
    Mixes continuous low-intensity background Poisson flows with
    heavy, periodic overlapping 'burst blocks' every 200 seconds.
    This guarantees both training AND test windows (last 20%) receive a 
    perfectly balanced mix of truly congested and purely normal recovery phases.
    """
    flows = []
    fid = 0
    t = 0.0
    
    # Background Traffic: continuous light Poisson flows
    while t < DURATION_S - 10:
        t += np.random.exponential(5)
        if t >= DURATION_S:
            break
        dur = float(np.clip(np.random.exponential(20), 5, 40))
        tgt = float(np.random.uniform(2, 6)) # low bandwidth
        flows.append((fid, t, dur, tgt))
        fid += 1

    # Burst Blocks: Periodic congestion every ~200 seconds
    # Leaves plenty of time for queues to drain back to normal between bursts
    burst_intervals = range(30, DURATION_S - 50, 200)
    for start_t in burst_intervals:
        # Create 3-5 heavy overlapping flows
        num_heavy = np.random.randint(3, 6)
        for i in range(num_heavy):
            offset = np.random.uniform(0, 10)
            dur = np.random.uniform(30, 80)
            tgt = np.random.uniform(8, 16) # high bandwidth
            flows.append((fid, start_t + offset, dur, tgt))
            fid += 1

    flows.sort(key=lambda x: x[1])
    return flows



# ════════════════════════════════════════════════════════════════════════════
# SECOND-BY-SECOND SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def simulate_network(flows: list) -> pd.DataFrame:
    records = []
    Q = 0.0
    retrans_cum = lost_cum = bytes_cum = 0

    for t in range(DURATION_S):
        active = [(fid, st, dur, tgt) for (fid, st, dur, tgt) in flows
                  if st <= t < st + dur]
        n_flows = len(active)
        total_tgt = sum(tgt for *_, tgt in active)

        rho = total_tgt / LINK_BW_MBPS

        # Expected queue from M/D/1 (saturated queue fills linearly above rho=1)
        if rho < 1.0:
            lq_exp = md1_lq(rho)
        else:
            lq_exp = MAX_QUEUE_PKTS * min((rho - 1.0) * 6 + 0.85, 1.0)

        # Add noise + EWMA inertia
        lq_exp = float(np.clip(lq_exp + np.random.normal(0, max(0.4, lq_exp * 0.15)), 0, MAX_QUEUE_PKTS + 4))
        Q = 0.35 * Q + 0.65 * lq_exp

        # Drops when queue overflows
        drops = 0
        if Q > MAX_QUEUE_PKTS:
            excess = Q - MAX_QUEUE_PKTS
            drops  = max(0, int(np.random.poisson(excess * max(1.0, rho) * 4)))
            Q      = float(MAX_QUEUE_PKTS) - np.random.uniform(0, 1.5)

        # Overlimits (HTB rate-limit events)
        if rho >= 1.0:
            overlimits = int(np.random.poisson((rho - 0.9) * 80 + 5))
        elif rho >= 0.85:
            overlimits = int(np.random.poisson((rho - 0.85) * 35))
        else:
            overlimits = 0

        # TCP retransmits (1 drop → ~1.2 retransmits, some spurious)
        retransmits = int(np.random.poisson(drops * 1.2)) if drops > 0 else 0
        lost_delta  = int(np.ceil(retransmits * np.random.uniform(0.3, 0.6)))

        # Queue depth (physical)
        q_pkts  = float(np.clip(Q, 0, MAX_QUEUE_PKTS))
        q_bytes = int(q_pkts * MTU_BYTES)

        # RTT and variance
        rtt_ms    = rtt_from_queue(q_pkts) + np.random.normal(0, max(0.4, q_pkts * 0.04))
        rtt_ms    = max(BASELINE_RTT_MS * 0.98, rtt_ms)
        rttvar_ms = (float(np.random.gamma(2, q_pkts * PKT_TX_MS / 2)) if q_pkts > 5
                     else float(np.random.uniform(0.3, 1.8)))

        # Achieved throughput (TCP AIMD penalty on drops)
        drop_factor    = 1.0 - min(0.65, drops * 0.06)
        achieved_total = min(total_tgt, LINK_BW_MBPS) * drop_factor if total_tgt > 0 else 0.0
        achieved_total = max(0.0, achieved_total + np.random.normal(0, 0.2))

        # Switch-side stats
        sent_pkt_delta = max(0, int(achieved_total * 1e6 / (MTU_BYTES * 8)))
        drop_ratio     = drops / max(sent_pkt_delta + drops, 1)

        # cwnd / ssthresh
        bdp_segs = max(4, int((LINK_BW_MBPS * 1e6 / 8 * rtt_ms / 1000) / MTU_BYTES))
        cwnd     = max(4, int(bdp_segs * drop_factor * np.random.uniform(0.7, 1.0)))
        ssthresh = max(cwnd, int(bdp_segs * 0.7))

        delivery_mbps = achieved_total * np.random.uniform(0.91, 1.02)

        # Cumulative host counters
        retrans_cum += retransmits
        lost_cum    += lost_delta
        bytes_delta  = max(0, int(achieved_total * 1e6 / 8))
        bytes_cum   += bytes_delta

        records.append({
            "t": t, "n_flows": n_flows, "active_flows": active,
            "total_tgt": total_tgt, "rho": rho,
            "q_pkts": q_pkts, "q_bytes": q_bytes,
            "drops": drops, "overlimits": overlimits,
            "retransmits": retransmits, "lost_delta": lost_delta,
            "rtt_ms": rtt_ms, "rttvar_ms": rttvar_ms,
            "achieved_total": achieved_total,
            "sent_pkt_delta": sent_pkt_delta, "drop_ratio": drop_ratio,
            "cwnd": cwnd, "ssthresh": ssthresh,
            "delivery_mbps": delivery_mbps,
            "retrans_cum": retrans_cum, "lost_cum": lost_cum,
            "bytes_cum": bytes_cum, "bytes_delta": bytes_delta,
        })

    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# CSV WRITERS
# ════════════════════════════════════════════════════════════════════════════

def write_iperf_csv(sim: pd.DataFrame, flows: list) -> None:
    rows = []
    for _, row in sim.iterrows():
        t      = int(row["t"])
        active = row["active_flows"]
        if not active:
            continue
        total_tgt = row["total_tgt"]
        for fid, st, dur, tgt in active:
            share    = tgt / max(total_tgt, 1e-9)
            mbps     = max(0.0, float(row["achieved_total"]) * share + np.random.normal(0, 0.1))
            retx     = max(0, int(round(row["retransmits"] * share)))
            rtt_us   = max(BASELINE_RTT_US * 0.95,
                           float(row["rtt_ms"]) * 1000 + np.random.normal(0, 200))
            rtv_us   = max(300, float(row["rttvar_ms"]) * 1000 + np.random.uniform(0, 300))
            rows.append({
                "flow_id":           fid,
                "absolute_time_s":   t + np.random.uniform(0, 0.99),
                "omitted":           False,
                "target_bw_mbps":    round(tgt, 2),
                "mbps_achieved":     round(mbps, 4),
                "retransmits":       retx,
                "rtt_us":            round(rtt_us, 1),
                "rttvar_us":         round(rtv_us, 1),
                "snd_cwnd_bytes":    int(row["cwnd"]) * MTU_BYTES,
                "bytes_transferred": max(0, int(mbps * 1e6 / 8)),
            })
    df   = pd.DataFrame(rows)
    path = os.path.join(IPERF_DIR, "iperf_timeseries.csv")
    df.to_csv(path, index=False)
    print(f"  iperf_timeseries.csv   : {len(df):5d} rows, "
          f"{df['flow_id'].nunique()} flows → {path}")


def write_ping_csv(sim: pd.DataFrame) -> None:
    rows = []
    seqs = [0, 0, 0]
    for _, row in sim.iterrows():
        t = int(row["t"])
        for sid in range(3):
            off = sid * 0.33 + np.random.uniform(-0.08, 0.08)
            rtt = max(BASELINE_RTT_MS * 0.95,
                      float(row["rtt_ms"]) + np.random.normal(0, max(0.4, float(row["rtt_ms"]) * 0.01)))
            rows.append({
                "stream_id":        sid,
                "unix_timestamp_s": round(UNIX_T0 + t + off, 3),
                "status":           "reply",
                "icmp_seq":         seqs[sid],
                "rtt_ms":           round(rtt, 2),
            })
            seqs[sid] += 1
    df   = pd.DataFrame(rows)
    path = os.path.join(IPERF_DIR, "ping_timeseries.csv")
    df.to_csv(path, index=False)
    print(f"  ping_timeseries.csv    : {len(df):5d} rows → {path}")


def write_qdisc_csv(sim: pd.DataFrame) -> None:
    rows = []
    for _, row in sim.iterrows():
        ts = float(row["t"]) + 0.5
        for iface in ["s1-eth1", "s1-eth2"]:
            j = np.random.uniform(0.92, 1.08)
            rows.append({
                "timestamp_s":          round(ts, 3),
                "interface":            iface,
                "backlog_packets":      max(0, int(row["q_pkts"] * j)),
                "backlog_bytes":        max(0, int(row["q_bytes"] * j)),
                "dropped_delta":        int(row["drops"]),
                "overlimits_delta":     int(row["overlimits"]),
                "throughput_mbps":      round(float(row["achieved_total"]) * j, 4),
                "link_utilization_pct": round(100.0 * float(row["achieved_total"]) / LINK_BW_MBPS * j, 2),
                "sent_packets_delta":   int(row["sent_pkt_delta"]),
                "overlimit_rate_pps":   round(float(row["overlimits"]) * j, 3),
                "drop_ratio":           round(float(row["drop_ratio"]), 6),
            })
    df   = pd.DataFrame(rows)
    path = os.path.join(SWITCH_DIR, "qdisc_metrics.csv")
    df.to_csv(path, index=False)
    print(f"  qdisc_metrics.csv      : {len(df):5d} rows → {path}")


def write_host_csv(sim: pd.DataFrame) -> None:
    rows = []
    for _, row in sim.iterrows():
        if int(row["n_flows"]) == 0:
            continue
        rtt = float(row["rtt_ms"])
        rows.append({
            "timestamp_s":         round(float(row["t"]) + 0.5, 3),
            "socket_count":        max(1, int(row["n_flows"])),
            "rtt_ms_mean":         round(rtt, 3),
            "rtt_ms_max":          round(rtt + np.random.uniform(0, 4), 3),
            "rttvar_ms_mean":      round(float(row["rttvar_ms"]), 3),
            "minrtt_ms_mean":      round(BASELINE_RTT_MS + np.random.uniform(0.0, 0.8), 3),
            "minrtt_ms_min":       round(BASELINE_RTT_MS + np.random.uniform(-0.3, 0.3), 3),
            "cwnd_segs_mean":      int(row["cwnd"]),
            "cwnd_segs_min":       max(4, int(row["cwnd"]) - np.random.randint(0, 4)),
            "ssthresh_mean":       int(row["ssthresh"]),
            "retrans_total_sum":   int(row["retrans_cum"]),
            "retrans_delta_sum":   int(row["retransmits"]),
            "lost_total_sum":      int(row["lost_cum"]),
            "lost_delta_sum":      int(row["lost_delta"]),
            "bytes_sent_sum":      int(row["bytes_cum"]),
            "bytes_sent_delta_sum":int(row["bytes_delta"]),
            "delivery_mbps_mean":  round(float(row["delivery_mbps"]), 4),
        })
    df   = pd.DataFrame(rows)
    path = os.path.join(HOST_DIR, "tcp_socket_metrics.csv")
    df.to_csv(path, index=False)
    print(f"  tcp_socket_metrics.csv : {len(df):5d} rows → {path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Synthetic Data Generator — SDN Congestion Experiment")
    print(f"Topology: 25Mbps, 30ms delay, {MAX_QUEUE_PKTS}-packet queue")
    print("=" * 60)

    print("\n[1/3] Generating flow schedule (Poisson arrivals)...")
    flows = generate_flow_schedule()
    active_secs = sum(1 for t in range(DURATION_S)
                      if any(f[1] <= t < f[1] + f[2] for f in flows))
    print(f"  Total flows     : {len(flows)}")
    print(f"  Active seconds  : {active_secs} / {DURATION_S}")

    print("\n[2/3] Simulating network second-by-second (M/D/1 + AIMD)...")
    sim = simulate_network(flows)
    print(f"  Mean utilization: {sim['rho'].mean():.2f}")
    print(f"  Has-drops secs  : {(sim['drops'] > 0).sum()}")
    print(f"  Max queue depth : {sim['q_pkts'].max():.1f} / {MAX_QUEUE_PKTS} pkts")
    print(f"  Mean RTT        : {sim['rtt_ms'].mean():.1f} ms (baseline={BASELINE_RTT_MS:.0f}ms)")

    print("\n[3/3] Writing CSV files...")
    write_iperf_csv(sim, flows)
    write_ping_csv(sim)
    write_qdisc_csv(sim)
    write_host_csv(sim)

    print("\n[Done] All CSVs written.")
    print("  Next: python dataset/build_dataset.py")


if __name__ == "__main__":
    main()
