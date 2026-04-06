# Congestion Detection in SDN Networks

## Project Overview
Detect network congestion in a Mininet SDN topology using tc-qdisc metrics,
iperf3 flow data, ICMP ping latency, and TCP socket stats. Labels are
generated using the **Congestion Evidence Framework (CEF)** — a physics-first
multi-layer consensus method where every threshold is derived analytically
from topology parameters. An ML classifier (RF / GBM / XGBoost) is trained
on the labeled dataset.

## Execution Environment
- **Data collection (topology/mytopo.py):** Linux only — requires Mininet + OVS kernel switch.
- **Dataset building + ML (dataset/build_dataset.py, models/train_model.py):** Any OS with Python 3.
- **Development/testing on Windows:** implement `dataset/generate_synthetic_data.py` to produce
  realistic synthetic CSVs in place of a real Mininet run (NOT YET IMPLEMENTED).

## File Structure

```
congestion-detection/
├── CLAUDE.md                          # This file (AI assistant context)
├── README.md                          # User-facing project overview
├── requirements.txt                   # Python dependencies
│
├── topology/
│   ├── mytopo.py                      # Mininet h1-s1-h2 topology + iperf/ping/qdisc
│   └── collect_switch_metrics.py      # tc-qdisc + OVS flow rate poller
│
├── collectors/
│   └── collect_host_metrics.py        # TCP socket stats via ss -tin (called by mytopo.py)
│
├── dataset/
│   └── build_dataset.py               # CEF labeling pipeline
│
├── models/
│   └── train_model.py                 # RF + GBM + XGBoost training + evaluation
│
└── notebooks/
    └── Visualize_iperf_data.ipynb     # Exploratory iperf/ping analysis

Output directories (created at runtime):
├── iperf_results/                     # iperf_timeseries.csv, ping_timeseries.csv
├── switch_results/                    # qdisc_metrics.csv, flow_metrics.csv
├── host_results/                      # tcp_socket_metrics.csv
├── dataset/                           # labeled_dataset.csv, cef_evidence.csv,
│                                      # labeling_report.txt, cef_timeline.png
└── model/                             # best_model.pkl, scaler.pkl,
                                       # evaluation_report.txt, plots
```

## Run Order

### On Linux (real Mininet data):
```bash
cd congestion-detection
sudo python3 topology/mytopo.py        # ~3.5 min — generates all 4 CSVs
python3 dataset/build_dataset.py       # CEF labeling → labeled_dataset.csv
python3 models/train_model.py          # ML training + evaluation → best_model.pkl
```

### On Windows (synthetic data — TODO):
```bash
# Once generate_synthetic_data.py is implemented:
python3 dataset/generate_synthetic_data.py  # generates realistic CSVs (no Mininet needed)
python3 dataset/build_dataset.py
python3 models/train_model.py
```

## Topology Parameters (topology/mytopo.py — do not change without updating dataset/build_dataset.py)

```
h1 (10.0.0.1) --[30ms, 25Mbps, queue=30]--> s1 --[30ms, 25Mbps, queue=30]--> h2 (10.0.0.2)
```

| Parameter | Value | Source in topology/mytopo.py |
|---|---|---|
| `LINK_BW_MBPS` | 25.0 | `bw=25` in `net.addLink` |
| `LINK_DELAY_MS` | 30.0 | `delay="30ms"` in `net.addLink` |
| `MAX_QUEUE_PKTS` | 30 | `max_queue_size=30` in `net.addLink` |
| `BASELINE_RTT_MS` | 120 | 4 × 30ms (h1→s1→h2→s1→h1) |

**Critical:** If topology parameters change, recalculate CEF thresholds in `dataset/build_dataset.py`.
They are analytically derived from these constants—arbitrary changes break the entire labeling framework.

## Labeling: Host-Flow Congestion Evidence Framework (HF-CEF)

**Core design principle:** Labels come exclusively from HOST-OBSERVABLE flow signals.
Switch qdisc metrics are collected but play NO role in labeling — they are ML features only.

### Why this separation matters
If switch data were used to label the dataset, a simple `if dropped_delta > 0: congested`
rule would be sufficient — no ML needed. Instead:
- **Labels** = what congestion *means to the end-host* (RTT inflation, retransmits, BW loss)
- **ML features** = what the SDN controller can *see at runtime* (switch qdisc counters)
- **ML learns** the mapping: switch observables → host-experienced congestion state

### Derived constants
- `PKT_TX_MS = (1500×8) / (25×10⁶) × 10³ = 0.48 ms`
- `T_RTT_RATIO = (120 + 23×0.48) / 120 = 1.092` — M/D/1 onset at ρ=0.75
- `T_RTTVAR_US = 5000 µs` — queuing jitter above OS baseline (~2ms)
- `T_BW_RATIO = 0.80` — TCP AIMD steady-state floor (Mathis et al. 1997)

### Evidence signals (ALL from iperf3 / ping / ss — NO switch data)

| Flag | Condition | Source | Justification |
|---|---|---|---|
| `hf_A_retx_iperf` | `iperf_retransmits_sum > 0` | iperf3 | Application-visible TCP loss recovery |
| `hf_A_retx_host` | `host_retrans_delta_sum > 0` | ss -tin | Kernel TCP retransmit counter |
| `hf_A_lost_host` | `host_lost_delta_sum > 0` | ss -tin | Kernel definitive lost-segment count |
| `hf_A_loss` | `(A_retx_iperf AND A_retx_host) OR A_lost_host` | Dual-source | Confirmed packet loss |
| `hf_B_rtt_iperf` | `iperf_rtt / baseline > 1.092` | iperf3 | RTT above M/D/1 onset |
| `hf_B_rtt_ping` | `ping_rtt / baseline > 1.092` | ICMP ping | Independent corroboration |
| `hf_B_rttvar` | `iperf_rttvar_us > 5000` | iperf3 | Queue jitter onset |
| `hf_C_bw_iperf` | `achieved_ratio < 0.80` | iperf3 | AIMD throughput degradation |
| `hf_C_delivery_ss` | `ss_delivery / link_bw < 0.80` | ss -tin | Kernel delivery rate drop |

### Label rules
- **Congested (2):** `hf_A_loss = True` — confirmed TCP packet loss at the host
- **Onset (1):** `hf_A_loss = False AND hf_B_latency = True AND hf_C_throughput = True`
- **Normal (0):** neither of the above
- **Binary:** Normal=0, Onset+Congested=1

### Do not
- Use ANY qdisc column (backlog, dropped_delta, overlimits) in label derivation
- Use GMM or unsupervised clustering — evaluators want physically-justified labels
- Change T_RTT_RATIO manually — recalculate from topology params if link delay/BW changes
- Use `hf_*` evidence columns as ML features — direct label derivations (data leakage)

## ML Pipeline (models/train_model.py)

- **Target:** `congestion_binary` (0=normal, 1=onset+congested)
- **Features:** ONLY `qdisc_*` switch metrics + temporal derivatives
  (rolling [3s, 10s, 30s] mean/std/max, lag [1,3,5s], z-score, rate-of-change, interactions)
- **Excluded from features:** all `hf_*`, `iperf_*`, `ping_*`, `host_*`, `rtt_relative_*`
  — these are label sources and unavailable to an SDN controller at runtime
- **Split:** time-aware at 80th percentile of `time_bin` — no random shuffle
- **Models:** RandomForest, GradientBoosting, XGBoost (ensemble comparison)
- **Best model selected by:** test F1-macro (robust to class imbalance)

## Key Design Decisions

1. **Labels from host/flow, features from switch (HF-CEF):** This is the core architectural
   decision. Labels represent *host-experienced congestion* (RTT inflation, TCP loss, BW
   degradation seen by iperf3/ping/ss). ML features are *switch-observable metrics* (qdisc
   backlog, drops, overlimits). This separation ensures the model does something useful:
   predict what hosts experience from what the controller can see — without host instrumentation.

2. **Why ML over if-else on switch data:** A static threshold (`if backlog > 23: congested`) is
   instantaneous, one-dimensional, and reactive. The ML model uses rolling-window temporal
   features to detect congestion trends before they produce drops, learns non-linear interactions
   between switch metrics, and generalizes across traffic patterns — all impossible with thresholds.

3. **Physics-first thresholds for labeling:** Every threshold is derived from topology parameters
   (M/D/1 queuing for T_RTT_RATIO, Mathis et al. AIMD analysis for T_BW_RATIO, queuing jitter
   analysis for T_RTTVAR). No threshold is chosen by intuition or fitted to data.

4. **Dual-source loss confirmation:** CONGESTED requires **both** iperf3 AND kernel ss to report
   retransmits (or ss to report lost segments). Single-source retransmit counts can be artifacts;
   dual-source agreement eliminates this class of false positives.

5. **Onset detection enables proactive SDN control:** Detecting congestion *before* packet drops
   gives the controller time to reroute or reduce ingress rate. Onset fires when RTT is elevated
   above the M/D/1 queue-growth knee AND throughput is already degraded — but no loss yet.

6. **Time-aware train/test split:** Random splitting leaks future network state into training
   (temporal data leakage). Always split by `time_bin` percentile to honor causality.

7. **Kernel RTT baseline (ss -tin):** `min_rtt` from ss is the minimum RTT observed since socket
   creation, measured before queue buildup — a data-derived baseline that replaces the analytical
   4×30ms calculation and further reduces assumptions in the labeling.

## Dependencies
See `requirements.txt`:
```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.5.0
```

**System requirements (Linux data collection only):**
- Mininet 2.3+ (virtual network emulator)
- Open vSwitch (OVS) kernel module
- iperf3 (traffic generation)
- `ss` and `tc` utilities (Linux coreutils)

Install Python deps: `pip install -r requirements.txt`

## Module Descriptions

### topology/mytopo.py
- Creates h1-s1-h2 topology with 30ms delay, 25Mbps BW, 30-packet queue per link
- Runs 200-second experiment with:
  - Poisson-distributed TCP flow arrivals (iperf3 with random BW/duration targets)
  - Overlapping ICMP ping streams at increasing frequency
  - HTB/netem qdisc with forced-drop behavior
- **Calls:** `collectors/collect_host_metrics.py` to capture TCP socket stats
- **Outputs:** `iperf_results/`, `switch_results/`, `host_results/`

### topology/collect_switch_metrics.py
- Polls `tc -s qdisc show` and OVS flow tables every second
- Extracts: dropped packets, backlog queue depth, overlimit events, throughput
- **Outputs:** `switch_results/qdisc_metrics.csv`, `switch_results/flow_metrics.csv`

### collectors/collect_host_metrics.py
- Polls `ss -tin state established` every second from h1
- Extracts TCP per-socket metrics: min_rtt, cwnd, ssthresh, retrans, lost, delivery_rate
- **Outputs:** `host_results/tcp_socket_metrics.csv`

### dataset/build_dataset.py
- Loads all 4 CSVs, aligns to 1-second bins with forward-fill for missing values
- Applies CEF labeling logic: derives `congestion_state` (0/1/2) based on physical thresholds
- Generates `dataset/labeled_dataset.csv` + diagnostic outputs
  - `cef_evidence.csv` — per-second evidence flags for audit/visualization
  - `labeling_report.txt` — documents every threshold calculation
  - `cef_timeline.png` — visualizes state transitions + evidence

### models/train_model.py
- Loads labeled dataset, engineers features (rolling windows, lags, z-scores, deltas)
- Converts 3-state `congestion_label` to binary (0=normal, 1=onset+congested)
- Trains RandomForest, GradientBoosting, and XGBoost (if available)
- Time-aware 80/20 train/test split on `time_bin` percentile
- Selects best model by test F1-macro (handles class imbalance)
- **Outputs:** `model/best_model.pkl`, `model/scaler.pkl`, `model/evaluation_report.txt`, plots

## Critical Rules (Do Not Break)

1. **Topology parity:** If you change BW, delay, or queue size in `topology/mytopo.py`,
   update the derived constants at the top of `dataset/build_dataset.py`:
   - `PKT_TX_MS`, `MAX_QUEUE_DELAY_MS`, `T1_BACKLOG_PKTS`, `T2_RTT_RATIO`, `T3_BW_RATIO`
   - Use the formulas documented in the CEF section above.

2. **No feature leakage:** Never include `cef_E_*`, `congestion_state`, `congestion_label`,
   `congestion_binary`, or `time_bin` as ML features — they are label derivations or splits.

3. **No GMM or unsupervised clustering:** Evaluators expect physically-justified labels.
   If CEF thresholds seem wrong, recalculate—don't add heuristics or statistical methods.

4. **Time-aware splits always:** Use `time_bin` percentile, never random shuffle.
   This respects causality and prevents data leakage from future to past.

5. **Multi-layer consensus is non-negotiable:** Require ≥2 independent protocol layers
   (queue, TCP, latency) before declaring congestion. It's the core defense against noise.

## TODO / Known Gaps

- **`dataset/generate_synthetic_data.py` not implemented:** Build this to enable Windows testing.
  Should generate realistic CSVs matching Mininet output format and distributions.
- **No real-time inference:** `best_model.pkl` is for batch evaluation only. For live SDN
  controller integration, wrap it in a streaming aggregator that maintains rolling windows.

## References

- M/D/1 queue analysis: Ivo Adan & Jacques Resing, "Queueing Theory"
- TCP AIMD steady-state: Mathis et al. (1997), "The macroscopic behavior of the TCP congestion avoidance algorithm"
- Mininet: Lantz, Heller, McKeown (2010), "A network in a laptop"
- Open vSwitch: Pfaff et al., "The Design and Implementation of Open vSwitch"
