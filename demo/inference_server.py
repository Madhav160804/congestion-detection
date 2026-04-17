#!/usr/bin/env python3
"""
SDN CongestionShield — Live Inference Server
=============================================
Starts a Mininet network, polls switch qdisc metrics every second,
runs the trained RandomForest model, and exposes a REST API + dashboard.

Run with:
    sudo python3 demo/inference_server.py
Then open: http://localhost:5050  in your Windows browser.
"""

import os, sys, re, time, json, pickle, threading, subprocess, collections
import numpy as np

# ── Ensure we can import from project root ──────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from flask import Flask, jsonify, send_from_directory, request
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel

# ── Constants ────────────────────────────────────────────────────────────────
LINK_BW_MBPS      = 5.0
POLL_INTERVAL     = 1.0        # seconds
HISTORY_LEN       = 120        # keep 120 seconds of history for dashboard
CONGESTION_THRESH = 3          # consecutive "congested" predictions → auto-response
RECOVERY_THRESH   = 5          # consecutive "normal" predictions → lift response
RATE_LIMIT_MBPS   = 2.0        # apply this rate cap when congested

MODEL_PATH   = os.path.join(ROOT, "model", "best_model.pkl")
SCALER_PATH  = os.path.join(ROOT, "model", "scaler.pkl")
FEAT_PATH    = os.path.join(ROOT, "model", "feature_names.txt")

# ── qdisc parsing (mirrors collect_switch_metrics.py) ───────────────────────
_QDISC_HDR_RE = re.compile(r'qdisc\s+(\S+)\s+(\S+):')
_SENT_RE      = re.compile(
    r'Sent\s+(\d+)\s+bytes\s+(\d+)\s+pkt\s+'
    r'\(dropped\s+(\d+),\s+overlimits\s+(\d+)'
)
_BACKLOG_RE   = re.compile(r'backlog\s+(\d+)b\s+(\d+)p')


def dump_qdisc(interface):
    raw = subprocess.run(
        f"tc -s qdisc show dev {interface}",
        shell=True, capture_output=True, text=True
    ).stdout
    qdiscs, current = [], None
    for line in raw.splitlines():
        hdr = _QDISC_HDR_RE.match(line.strip())
        if hdr:
            current = dict(interface=interface, qdisc_kind=hdr.group(1),
                           handle=hdr.group(2), sent_bytes=0, sent_packets=0,
                           dropped_packets=0, overlimits=0,
                           backlog_bytes=0, backlog_packets=0)
            qdiscs.append(current); continue
        if current is None: continue
        m = _SENT_RE.search(line)
        if m:
            current["sent_bytes"]      = int(m.group(1))
            current["sent_packets"]    = int(m.group(2))
            current["dropped_packets"] = int(m.group(3))
            current["overlimits"]      = int(m.group(4))
        m = _BACKLOG_RE.search(line)
        if m:
            current["backlog_bytes"]   = int(m.group(1))
            current["backlog_packets"] = int(m.group(2))
    return qdiscs


def aggregate_interfaces(snapshots_by_intf, prev_by_intf, dt):
    """Aggregate raw qdisc counters across all switch interfaces → one row."""
    tot_sent_bytes = tot_sent_pkt = tot_drop = tot_over = 0
    max_backlog_pkt = max_backlog_bytes = 0

    for intf, snaps in snapshots_by_intf.items():
        prev = prev_by_intf.get(intf, {})
        for snap in snaps:
            key = (intf, snap["handle"])
            p   = prev.get(key, snap)
            tot_sent_bytes  += max(0, snap["sent_bytes"]      - p.get("sent_bytes", 0))
            tot_sent_pkt    += max(0, snap["sent_packets"]    - p.get("sent_packets", 0))
            tot_drop        += max(0, snap["dropped_packets"] - p.get("dropped_packets", 0))
            tot_over        += max(0, snap["overlimits"]      - p.get("overlimits", 0))
            max_backlog_pkt  = max(max_backlog_pkt,  snap["backlog_packets"])
            max_backlog_bytes = max(max_backlog_bytes, snap["backlog_bytes"])

    if dt <= 0: dt = POLL_INTERVAL
    throughput = (tot_sent_bytes * 8) / (dt * 1e6)
    util_pct   = 100.0 * throughput / LINK_BW_MBPS
    over_rate  = tot_over / dt
    attempted  = tot_sent_pkt + tot_drop
    drop_ratio = tot_drop / attempted if attempted > 0 else 0.0

    return {
        "throughput_mbps":      round(throughput, 4),
        "util_pct":             round(util_pct, 2),
        "overlimit_delta":      tot_over,
        "overlimit_rate":       round(over_rate, 4),
        "dropped_delta":        tot_drop,
        "drop_ratio":           round(drop_ratio, 6),
        "sent_pkt_delta":       tot_sent_pkt,
        "backlog_pkt_max":      max_backlog_pkt,
        "backlog_bytes_max":    max_backlog_bytes,
    }


# ── Feature engineering (matches train_model.py) ────────────────────────────
#  Raw metric key → CSV column stem mapping
RAW_TO_STEM = {
    "util_pct":          "qdisc_util_pct_max",
    "throughput_mbps":   "qdisc_throughput_mbps_sum",
    "overlimit_rate":    "qdisc_overlimit_rate_sum",
    "overlimit_delta":   "qdisc_overlimit_delta_sum",
    "sent_pkt_delta":    "qdisc_sent_pkt_delta_sum",
}

FEATURE_NAMES = open(FEAT_PATH).read().splitlines()  # 15 features, exact order


def compute_features(history: collections.deque) -> np.ndarray:
    """
    Given a deque of raw metric dicts (up to 30 entries, 1 per second),
    compute the 15 engineered features the RF expects.
    """
    if len(history) < 2:
        return None  # Not enough history yet

    # Build per-stem arrays (raw values in time order)
    series = {stem: np.array([r[raw] for r in history], dtype=float)
              for raw, stem in RAW_TO_STEM.items()}

    def roll_stat(stem, window, stat):
        arr = series[stem]
        w   = min(window, len(arr))
        seg = arr[-w:]
        return float(np.max(seg) if stat == "max" else np.mean(seg))

    feat_map = {}
    for stem in series:
        for w in (10, 30):
            feat_map[f"{stem}_roll{w}_mean"] = roll_stat(stem, w, "mean")
            feat_map[f"{stem}_roll{w}_max"]  = roll_stat(stem, w, "max")

    # Return in exact order saved during training
    try:
        return np.array([feat_map[f] for f in FEATURE_NAMES], dtype=float)
    except KeyError as e:
        print(f"[Feature] Missing: {e}")
        return None


# ── Global state ─────────────────────────────────────────────────────────────
net            = None          # Mininet instance
h1 = h2 = s1  = None
interfaces     = []

model          = pickle.load(open(MODEL_PATH,  "rb"))
scaler         = pickle.load(open(SCALER_PATH, "rb"))

history        = collections.deque(maxlen=30)   # raw metrics
api_history    = collections.deque(maxlen=HISTORY_LEN)  # for dashboard
incidents      = []            # incident log entries
current_state  = {
    "prediction":   "NORMAL",
    "confidence":   1.0,
    "util_pct":     0.0,
    "throughput":   0.0,
    "overlimit_rate": 0.0,
    "rate_limited": False,
    "t":            0,
}

_consec_congested = 0
_consec_normal    = 0
_rate_limited     = False
_iperf_procs      = []       # active iperf3 client processes
_t_start          = None

state_lock = threading.Lock()


# ── Auto-response ────────────────────────────────────────────────────────────
def apply_rate_limit():
    global _rate_limited
    if _rate_limited or s1 is None:
        return
    try:
        for intf in interfaces:
            s1.cmd(
                f"tc qdisc del dev {intf} root 2>/dev/null; "
                f"tc qdisc add dev {intf} root tbf rate {RATE_LIMIT_MBPS}mbit "
                f"burst 32kbit latency 100ms"
            )
        _rate_limited = True
        msg = f"Auto-response: rate limit {RATE_LIMIT_MBPS} Mbps applied"
        print(f"[Response] {msg}")
        _log_incident("AUTO-RESPONSE", msg)
    except Exception as e:
        print(f"[Response] Error applying rate limit: {e}")


def lift_rate_limit():
    global _rate_limited
    if not _rate_limited or s1 is None:
        return
    try:
        for intf in interfaces:
            s1.cmd(f"tc qdisc del dev {intf} root 2>/dev/null")
        _rate_limited = False
        msg = "Auto-response: rate limit removed — network recovered"
        print(f"[Recovery] {msg}")
        _log_incident("RECOVERY", msg)
    except Exception as e:
        print(f"[Recovery] Error removing rate limit: {e}")


def _log_incident(kind, msg):
    incidents.append({
        "t":    round(time.time() - _t_start, 1),
        "kind": kind,
        "msg":  msg,
    })


# ── Polling loop ─────────────────────────────────────────────────────────────
_stop_poll = threading.Event()


def poll_loop():
    global _consec_congested, _consec_normal, _t_start

    prev_by_intf = {}
    prev_time    = None
    _t_start     = time.time()

    while not _stop_poll.is_set():
        t0 = time.time()
        ts = t0 - _t_start

        # ── Collect raw qdisc stats ─────────────────────────────────────
        snaps = {}
        for intf in interfaces:
            snaps[intf] = dump_qdisc(intf)

        dt  = (t0 - prev_time) if prev_time is not None else POLL_INTERVAL
        row = aggregate_interfaces(snaps, prev_by_intf, dt)

        # Update prev
        for intf, snap_list in snaps.items():
            prev_by_intf[intf] = {
                (intf, s["handle"]): s for s in snap_list
            }
        prev_time = t0

        history.append(row)

        # ── Compute features & predict ──────────────────────────────────
        feats = compute_features(history)
        pred_label = "NORMAL"
        confidence = 1.0

        if feats is not None and not np.any(np.isnan(feats)):
            feats_scaled = scaler.transform(feats.reshape(1, -1))
            proba = model.predict_proba(feats_scaled)[0]
            pred_class = int(np.argmax(proba))
            confidence = float(proba[pred_class])
            pred_label = ["NORMAL", "CONGESTED"][pred_class]

        # ── State machine ───────────────────────────────────────────────
        if pred_label == "CONGESTED":
            _consec_congested += 1
            _consec_normal = 0
        else:
            _consec_normal += 1
            _consec_congested = 0

        if _consec_congested >= CONGESTION_THRESH and not _rate_limited:
            apply_rate_limit()
        if _consec_normal >= RECOVERY_THRESH and _rate_limited:
            lift_rate_limit()

        # ── Update shared state ─────────────────────────────────────────
        with state_lock:
            current_state.update({
                "prediction":     pred_label,
                "confidence":     round(confidence, 4),
                "util_pct":       row["util_pct"],
                "throughput":     row["throughput_mbps"],
                "overlimit_rate": row["overlimit_rate"],
                "dropped_delta":  row["dropped_delta"],
                "backlog_pkt":    row["backlog_pkt_max"],
                "rate_limited":   _rate_limited,
                "t":              round(ts, 1),
            })
            api_history.append(dict(current_state))

        if pred_label == "CONGESTED":
            _log_incident("CONGESTED",
                f"util={row['util_pct']:.0f}% "
                f"overlimit={row['overlimit_rate']:.1f}/s "
                f"conf={confidence:.0%}")

        elapsed = time.time() - t0
        _stop_poll.wait(timeout=max(0, POLL_INTERVAL - elapsed))


# ── Traffic injection ─────────────────────────────────────────────────────────
def inject_traffic(n_flows=8):
    if h1 is None or h2 is None:
        return False
    h2_ip = h2.IP()
    port_base = 5201

    # Start iperf3 servers on h2
    for i in range(n_flows):
        h2.cmd(f"iperf3 -s -p {port_base+i} -D 2>/dev/null")

    # Start iperf3 clients on h1
    for i in range(n_flows):
        h1.cmd(
            f"iperf3 -c {h2_ip} -p {port_base+i} "
            f"-b 2M -t 60 -J -i 1 "
            f"> /tmp/iperf_inject_{i}.json 2>&1 &"
        )
    _log_incident("INJECT", f"Traffic surge: {n_flows} flows launched (~{n_flows*2} Mbps)")
    return True


def clear_traffic():
    if h1 is None:
        return False
    h1.cmd("pkill -9 iperf3 2>/dev/null; true")
    h2.cmd("pkill -9 iperf3 2>/dev/null; true")
    _log_incident("CLEAR", "Traffic cleared — flows stopped")
    return True


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "dashboard"))


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state")
def api_state():
    with state_lock:
        s = dict(current_state)
    s["incidents"] = list(incidents)[-30:]   # last 30 events
    s["rate_limited"] = _rate_limited
    return jsonify(s)


@app.route("/api/history")
def api_history_route():
    with state_lock:
        hist = list(api_history)
    return jsonify(hist)


@app.route("/api/inject", methods=["POST"])
def api_inject():
    n = int(request.args.get("flows", 8))
    ok = inject_traffic(n)
    return jsonify({"ok": ok, "flows": n})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    ok = clear_traffic()
    return jsonify({"ok": ok})


# ── Mininet startup ───────────────────────────────────────────────────────────
def start_mininet():
    global net, h1, h2, s1, interfaces
    setLogLevel("warning")

    net = Mininet(switch=OVSSwitch, link=TCLink, controller=None)
    h1  = net.addHost("h1")
    h2  = net.addHost("h2")
    s1  = net.addSwitch("s1", failMode="standalone")
    net.addLink(h1, s1, bw=LINK_BW_MBPS, delay="5ms", max_queue_size=5)
    net.addLink(s1, h2, bw=LINK_BW_MBPS, delay="5ms", max_queue_size=5)
    net.start()

    # Disable TSO/GSO so HTB shaper sees individual packets
    for host in (h1, h2, s1):
        for intf in host.intfNames():
            host.cmd(f"ethtool -K {intf} tso off gso off gro off 2>/dev/null || true")

    # Switch to TCP Reno for faster drop response
    for host in (h1, h2):
        host.cmd("sysctl -w net.ipv4.tcp_congestion_control=reno 2>/dev/null")

    # Seed a steady baseline traffic (2 light flows ≈ 2 Mbps)
    h2_ip = h2.IP()
    for p in (5101, 5102):
        h2.cmd(f"iperf3 -s -p {p} -D 2>/dev/null")
    for p, bw in ((5101, "1M"), (5102, "1M")):
        h1.cmd(f"iperf3 -c {h2_ip} -p {p} -b {bw} -t 600 -J -i 1 > /dev/null 2>&1 &")

    # Discover switch interfaces (s1-eth1, s1-eth2)
    result = subprocess.run(
        "ovs-vsctl list-ports s1", shell=True, capture_output=True, text=True
    )
    interfaces[:] = [p.strip() for p in result.stdout.splitlines() if p.strip()]
    print(f"[Mininet] Started. Interfaces: {interfaces}")


def shutdown():
    _stop_poll.set()
    clear_traffic()
    if net:
        net.stop()
    print("[Mininet] Stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.geteuid() != 0:
        print("ERROR: Run as root →  sudo python3 demo/inference_server.py")
        sys.exit(1)

    print("=" * 55)
    print("  SDN CongestionShield — Inference Server")
    print("=" * 55)

    # Load model
    print(f"[Model] Loaded: {MODEL_PATH}")
    print(f"[Model] Features: {len(FEATURE_NAMES)}")

    # Start Mininet
    print("[Mininet] Initialising topology…")
    start_mininet()

    # Start polling thread
    poll_thread = threading.Thread(target=poll_loop, daemon=True)
    poll_thread.start()
    print("[Poll]   Metric collection started (1s interval)")

    # Start Flask
    print("[Flask]  Dashboard: http://localhost:5050")
    print("         Controls:  http://localhost:5050/api/inject?flows=8")
    print("         State:     http://localhost:5050/api/state")
    print("-" * 55)

    try:
        # Bind 0.0.0.0 so Windows browser can connect via localhost
        app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()
