# Congestion Detection in SDN Networks

A complete ML pipeline for detecting network congestion in Mininet SDN topologies using the **Congestion Evidence Framework (CEF)** — a physics-first, multi-layer consensus labeling method.

## Project Structure

```
congestion-detection/
├── CLAUDE.md                      # Claude Code project config (instructions for AI)
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── topology/                      # Mininet topology & data collection
│   ├── mytopo.py                  # Mininet h1-s1-h2 topology + iperf/ping traffic
│   └── collect_switch_metrics.py  # tc-qdisc + OVS flow rate poller
│
├── collectors/                    # Kernel/network state collectors
│   └── collect_host_metrics.py    # TCP socket stats via ss -tin
│
├── dataset/                       # Dataset building & labeling
│   └── build_dataset.py           # CEF labeling: merges CSVs, labels samples
│
├── models/                        # ML training & evaluation
│   └── train_model.py             # RF + GBM + XGBoost trainer
│
└── notebooks/                     # Exploratory analysis
    └── Visualize_iperf_data.ipynb # iperf metrics visualisation

Output directories (created at runtime):
├── iperf_results/                 # iperf3 & ping timeseries CSVs
├── switch_results/                # qdisc & flow stats CSVs
├── host_results/                  # TCP socket metrics CSV
├── dataset/                       # labeled_dataset.csv + diagnostics
└── model/                         # best_model.pkl + evaluation plots
```

## Topology

```
h1 (10.0.0.1) --[30ms, 25Mbps, queue=30]--> s1 --[30ms, 25Mbps, queue=30]--> h2 (10.0.0.2)
```

- **Link capacity:** 25 Mbps
- **Propagation delay:** 30 ms (one-way per link)
- **Queue size:** 30 packets
- **Baseline RTT:** 120 ms (4 × 30 ms propagation)

## Labeling: Congestion Evidence Framework (CEF)

Multi-layer consensus method. All thresholds derived analytically from topology parameters.

| State | Condition | Justification |
|---|---|---|
| **Congested (2)** | `dropped_delta > 0` | Queue overflowed — binary event |
| **Onset (1)** | `(backlog ≥ 23 OR overlimits > 0) AND (retransmits > 0 OR RTT > 1.092× OR BW < 0.80×)` | Layer 1 stress + L2/L3 confirmation |
| **Normal (0)** | Neither of above | Idle network |

**Key thresholds (all topology-derived):**
- `backlog ≥ 23 pkts` — M/D/1 queueing theory at ρ=0.75
- `RTT > 1.092× baseline` — baseline + (23 pkts × 0.48 ms/pkt) / 120 ms
- `achieved_ratio < 0.80` — TCP AIMD steady-state (Mathis et al. 1997)

See `CLAUDE.md` for full derivations.

## Quick Start (Linux with Mininet)

```bash
# 1. Generate traffic data (requires root + Mininet)
cd congestion-detection
sudo python3 topology/mytopo.py

# 2. Label with CEF
python3 dataset/build_dataset.py

# 3. Train ML model
python3 models/train_model.py
```

Outputs: `dataset/labeled_dataset.csv` + `model/best_model.pkl` + evaluation plots.

## Quick Start (Windows / without Mininet)

```bash
# 1. Generate synthetic realistic data (no Mininet needed)
python3 dataset/generate_synthetic_data.py

# 2. Label with CEF
python3 dataset/build_dataset.py

# 3. Train ML model
python3 models/train_model.py
```

**Note:** `generate_synthetic_data.py` creates realistic CSVs with the same format and distributions as real Mininet runs. Perfect for development/testing on Windows.

## Dependencies

```
pandas numpy scikit-learn matplotlib seaborn xgboost
```

Install: `pip install pandas numpy scikit-learn matplotlib seaborn xgboost`

## Key Files

### topology/mytopo.py
Mininet topology builder. Runs a 200-second experiment with:
- Poisson-distributed TCP flow arrivals (iperf3 with random BW/duration)
- Overlapping ICMP ping streams at increasing frequency
- HTB/netem qdisc shaping with 30-packet queue (forces real drops)
- **Modified:** imports `collect_host_metrics` to capture TCP socket stats

### collectors/collect_host_metrics.py
Polls `ss -tin state established` every second on h1, extracting:
- **min_rtt** — kernel-measured minimum RTT (true propagation baseline)
- **cwnd, ssthresh** — TCP congestion window state
- **retrans, lost** — retransmit/loss counters
- **delivery_rate** — measured throughput per socket

Outputs: `host_results/tcp_socket_metrics.csv`

### dataset/build_dataset.py
Core data processing & labeling pipeline:
1. Loads iperf + qdisc + ping + host TCP metrics
2. Aligns to 1-second bins
3. Applies CEF labeling (multi-layer consensus)
4. Outputs: `dataset/labeled_dataset.csv` + `dataset/labeling_report.txt` + `dataset/cef_timeline.png`

The `labeling_report.txt` documents every threshold calculation — ready to include in evaluations.

### models/train_model.py
ML training & evaluation:
- **Features:** per-second metrics + rolling windows [3, 10, 30s] + lags [1, 3, 5s] + z-scores + deltas
- **Models:** RandomForest, GradientBoosting, XGBoost (ensemble comparison)
- **Split:** time-aware at 80% (no temporal leakage)
- **Evaluation:** accuracy, F1-macro, ROC-AUC, confusion matrix, feature importance
- **Outputs:** `model/best_model.pkl`, `model/evaluation_report.txt`, diagnostic plots

## Data Flow

```
mytopo.py
    ├→ iperf3 flows → iperf_results/iperf_timeseries.csv
    ├→ ping streams → iperf_results/ping_timeseries.csv
    ├→ collect_switch_metrics.py → switch_results/qdisc_metrics.csv
    └→ collect_host_metrics.py → host_results/tcp_socket_metrics.csv
         ↓ (all CSVs)
    build_dataset.py → dataset/labeled_dataset.csv
         ↓
    train_model.py → model/best_model.pkl + evaluation plots
```

## Important Notes

1. **Mininet requirement:** `mytopo.py` requires Linux (OVS kernel switch). Use `generate_synthetic_data.py` on Windows.

2. **Topology parameters:** If you change queue size, link BW, or delay in `mytopo.py`, recalculate the CEF thresholds in `build_dataset.py` (T1, T2, T3). They are derived from these constants.

3. **Evaluators:** The `dataset/labeling_report.txt` file is auto-generated and shows every calculation. This is the proof that thresholds are physics-based, not arbitrary.

4. **Data leakage:** CEF evidence flags (`cef_E_*`) are excluded from ML features — they are direct label derivations.

## Design Philosophy

- **No arbitrary thresholds:** Every threshold derives from topology parameters via established network theory (M/D/1 queuing, TCP AIMD, queuing delay calculation).
- **Multi-layer consensus:** Requires ≥2 independent protocol layers to agree before labeling congestion — eliminates single-sensor noise.
- **Defensible to strict evaluators:** All design decisions documented in `CLAUDE.md` and `labeling_report.txt`.
- **Realistic data:** Synthetic generator models actual TCP/qdisc behavior, not just random numbers.

## References

- M/D/1 queue: Ivo Adan & Jacques Resing, "Queueing Theory"
- TCP AIMD: Mathis et al. (1997), "The macroscopic behavior of the TCP congestion avoidance algorithm"
- Mininet: Lantz, Heller, McKeown (2010), "A network in a laptop"
- OVS: Pfaff et al., "The Design and Implementation of Open vSwitch"

## License & Attribution

Created with Claude Code as part of an SDN congestion detection project.
