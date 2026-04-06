#!/usr/bin/env python3
"""
Link-Layer Congestion Metrics Collector
=========================================
Polls tc qdisc statistics from the veth interfaces attached to the Mininet
switch, plus OVS flow-table counters, and writes two timeseries CSVs:

  switch_results/qdisc_metrics.csv  -- per-interface qdisc stats (drops, backlog,
                                       throughput, utilisation) -- PRIMARY SOURCE
  switch_results/flow_metrics.csv   -- per OVS flow-entry byte/packet rates

Why tc qdiscs and not ovs-ofctl dump-ports?
--------------------------------------------
  OVS is a pure software switch inside the kernel. There is no physical NIC,
  so frame errors, CRC errors, collisions and RX overruns are always 0.
  OVS RX counters are 0 because forwarding happens inside the OVS datapath
  before the packet reaches the destination veth.
  Actual packet drops from bandwidth shaping happen at the Linux qdisc layer
  (tc HTB + netem), which OVS has no visibility into.
  tc -s qdisc show dev <intf> exposes the real drop counters, queue backlog,
  and byte/packet totals directly from the qdisc where shaping occurs.

Key metrics for congestion detection
--------------------------------------
  backlog_packets    packets currently queued -- most direct congestion signal
  backlog_bytes      bytes currently queued
  dropped_delta      packets dropped this interval (queue overflow)
  overlimits_delta   packets that hit the rate limit (rises BEFORE drops)
  drop_ratio         drops / (sent + drops) -- normalised congestion intensity
  throughput_mbps    bytes sent this interval converted to Mbps
  link_utilization_pct  throughput / link_capacity * 100

Usage (standalone):
    sudo python3 collect_switch_metrics.py --switch s1 --interval 1 --duration 180

Usage (imported):
    from collect_switch_metrics import start_switch_monitor, stop_switch_monitor
    monitor = start_switch_monitor(net.get('s1'), interval=1.0, link_bw_mbps=10.0)
    run_test(net)
    stop_switch_monitor(monitor)
"""

import re
import os
import csv
import time
import argparse
import threading
import subprocess


RESULTS_DIR = "switch_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

QDISC_CSV = os.path.join(RESULTS_DIR, "qdisc_metrics.csv")
FLOW_CSV  = os.path.join(RESULTS_DIR, "flow_metrics.csv")


# ---------------------------------------------------------------------------
# 1.  DISCOVER SWITCH INTERFACES
# ---------------------------------------------------------------------------

def get_switch_interfaces(switch_name):
    result = subprocess.run(
        "ovs-vsctl list-ports " + switch_name,
        shell=True, capture_output=True, text=True
    )
    return [p.strip() for p in result.stdout.splitlines() if p.strip()]


# ---------------------------------------------------------------------------
# 2.  TC QDISC PARSER
# ---------------------------------------------------------------------------
#
# tc -s qdisc show dev <intf> produces blocks like:
#
#   qdisc htb 1: root refcnt 2 r2q 10 default 0x1 direct_packets_stat 0
#    Sent 123456 bytes 890 pkt (dropped 12, overlimits 45 requeues 0)
#    backlog 9876b 7p requeues 0
#
#   qdisc netem 10: parent 1:1 limit 15
#    Sent 123456 bytes 890 pkt (dropped 5, overlimits 0 requeues 0)
#    backlog 1500b 1p requeues 0
#
# We collect every qdisc layer so nothing is missed.

_QDISC_HDR_RE = re.compile(r'qdisc\s+(\S+)\s+(\S+):')
_SENT_RE      = re.compile(
    r'Sent\s+(\d+)\s+bytes\s+(\d+)\s+pkt\s+'
    r'\(dropped\s+(\d+),\s+overlimits\s+(\d+)'
)
_BACKLOG_RE   = re.compile(r'backlog\s+(\d+)b\s+(\d+)p')


def dump_qdisc(interface):
    raw = subprocess.run(
        "tc -s qdisc show dev " + interface,
        shell=True, capture_output=True, text=True
    ).stdout

    qdiscs  = []
    current = None

    for line in raw.splitlines():
        hdr = _QDISC_HDR_RE.match(line.strip())
        if hdr:
            current = {
                "interface":        interface,
                "qdisc_kind":       hdr.group(1),
                "handle":           hdr.group(2),
                "sent_bytes":       0,
                "sent_packets":     0,
                "dropped_packets":  0,
                "overlimits":       0,
                "backlog_bytes":    0,
                "backlog_packets":  0,
            }
            qdiscs.append(current)
            continue

        if current is None:
            continue

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


# ---------------------------------------------------------------------------
# 3.  OVS FLOW TABLE PARSER
# ---------------------------------------------------------------------------

_FLOW_RE = re.compile(
    r'cookie=(0x[\da-fA-F]+).*?'
    r'duration=([\d.]+)s.*?'
    r'table=(\d+).*?'
    r'n_packets=(\d+).*?'
    r'n_bytes=(\d+).*?'
    r'priority=(\d+)'
    r'(.*?)actions=(\S+)'
)


def dump_flows(switch_name):
    raw = subprocess.run(
        "ovs-ofctl dump-flows " + switch_name,
        shell=True, capture_output=True, text=True
    ).stdout
    flows = []
    for line in raw.splitlines():
        m = _FLOW_RE.search(line)
        if m:
            flows.append({
                "cookie":       m.group(1),
                "duration_sec": float(m.group(2)),
                "table":        int(m.group(3)),
                "n_packets":    int(m.group(4)),
                "n_bytes":      int(m.group(5)),
                "priority":     int(m.group(6)),
                "match":        m.group(7).strip().strip(","),
                "actions":      m.group(8),
            })
    return flows


# ---------------------------------------------------------------------------
# 4.  DERIVE RATES FROM CONSECUTIVE SNAPSHOTS
# ---------------------------------------------------------------------------

def _derive_qdisc_row(ts, curr, prev, dt, link_bw_mbps):

    def d(key):
        return max(0, curr.get(key, 0) - prev.get(key, 0))

    sent_pkt_d  = d("sent_packets")
    sent_byte_d = d("sent_bytes")
    drop_d      = d("dropped_packets")
    over_d      = d("overlimits")

    throughput  = (sent_byte_d * 8) / (dt * 1e6) if dt > 0 else 0.0
    utilization = 100.0 * throughput / link_bw_mbps if link_bw_mbps > 0 else 0.0
    drop_rate   = drop_d / dt if dt > 0 else 0.0
    over_rate   = over_d / dt if dt > 0 else 0.0
    attempted   = sent_pkt_d + drop_d
    drop_ratio  = drop_d / attempted if attempted > 0 else 0.0

    return {
        # identity
        "timestamp_s":          round(ts, 3),
        "interface":            curr["interface"],
        "qdisc_kind":           curr["qdisc_kind"],
        "handle":               curr["handle"],
        "poll_interval_s":      round(dt, 3),
        # cumulative totals (since switch start)
        "sent_bytes_total":     curr["sent_bytes"],
        "sent_packets_total":   curr["sent_packets"],
        "dropped_total":        curr["dropped_packets"],
        "overlimits_total":     curr["overlimits"],
        # instantaneous queue depth -- most direct congestion indicator
        "backlog_bytes":        curr["backlog_bytes"],
        "backlog_packets":      curr["backlog_packets"],
        # per-interval deltas
        "sent_bytes_delta":     sent_byte_d,
        "sent_packets_delta":   sent_pkt_d,
        "dropped_delta":        drop_d,
        "overlimits_delta":     over_d,
        # derived rates
        "throughput_mbps":      round(throughput,  4),
        "link_utilization_pct": round(utilization, 2),
        "drop_rate_pps":        round(drop_rate,   4),
        "overlimit_rate_pps":   round(over_rate,   4),
        "drop_ratio":           round(drop_ratio,  6),
    }


# ---------------------------------------------------------------------------
# 5.  POLLING LOOP
# ---------------------------------------------------------------------------

QDISC_FIELDNAMES = [
    "timestamp_s", "interface", "qdisc_kind", "handle", "poll_interval_s",
    "sent_bytes_total", "sent_packets_total", "dropped_total", "overlimits_total",
    "backlog_bytes", "backlog_packets",
    "sent_bytes_delta", "sent_packets_delta", "dropped_delta", "overlimits_delta",
    "throughput_mbps", "link_utilization_pct",
    "drop_rate_pps", "overlimit_rate_pps", "drop_ratio",
]

FLOW_FIELDNAMES = [
    "timestamp_s", "switch", "flow_index",
    "cookie", "duration_sec", "table", "priority",
    "n_packets", "n_bytes", "match", "actions",
    "n_packets_delta", "n_bytes_delta", "byte_rate_mbps",
]


def _poll_loop(switch_name, interfaces, interval, link_bw_mbps, stop_event, t_start):

    with open(QDISC_CSV, "w", newline="") as qf, \
         open(FLOW_CSV,  "w", newline="") as ff:

        qdisc_writer = csv.DictWriter(qf, fieldnames=QDISC_FIELDNAMES)
        flow_writer  = csv.DictWriter(ff, fieldnames=FLOW_FIELDNAMES)
        qdisc_writer.writeheader()
        flow_writer.writeheader()

        prev_qdiscs = {}   # (interface, handle) -> last snapshot
        prev_flows  = {}   # flow_index -> {n_packets, n_bytes}
        prev_time   = None

        while not stop_event.is_set():
            poll_start = time.time()
            ts         = poll_start - t_start
            dt         = (poll_start - prev_time) if prev_time is not None else interval

            # -- qdisc stats -----------------------------------------------
            for intf in interfaces:
                for curr in dump_qdisc(intf):
                    key  = (intf, curr["handle"])
                    prev = prev_qdiscs.get(key, curr)
                    qdisc_writer.writerow(
                        _derive_qdisc_row(ts, curr, prev, dt, link_bw_mbps)
                    )
                    prev_qdiscs[key] = curr
            qf.flush()

            # -- flow table stats ------------------------------------------
            for idx, flow in enumerate(dump_flows(switch_name)):
                prev_f    = prev_flows.get(idx, {})
                pkt_delta = flow["n_packets"] - prev_f.get("n_packets", flow["n_packets"])
                byt_delta = flow["n_bytes"]   - prev_f.get("n_bytes",   flow["n_bytes"])
                byt_rate  = (byt_delta * 8) / (dt * 1e6) if dt > 0 else 0.0
                flow_writer.writerow({
                    "timestamp_s":     round(ts, 3),
                    "switch":          switch_name,
                    "flow_index":      idx,
                    "cookie":          flow["cookie"],
                    "duration_sec":    flow["duration_sec"],
                    "table":           flow["table"],
                    "priority":        flow["priority"],
                    "n_packets":       flow["n_packets"],
                    "n_bytes":         flow["n_bytes"],
                    "match":           flow["match"],
                    "actions":         flow["actions"],
                    "n_packets_delta": max(0, pkt_delta),
                    "n_bytes_delta":   max(0, byt_delta),
                    "byte_rate_mbps":  round(byt_rate, 4),
                })
                prev_flows[idx] = {
                    "n_packets": flow["n_packets"],
                    "n_bytes":   flow["n_bytes"],
                }
            ff.flush()

            prev_time = poll_start
            stop_event.wait(timeout=max(0.0, interval - (time.time() - poll_start)))


# ---------------------------------------------------------------------------
# 6.  PUBLIC API
# ---------------------------------------------------------------------------

def start_switch_monitor(switch, interval=1.0, link_bw_mbps=10.0, t_start=None):
    """
    Start a background thread that polls tc qdisc + OVS flow stats.

    Parameters
    ----------
    switch       : Mininet switch object or switch name string
    interval     : polling interval in seconds (default 1.0)
    link_bw_mbps : link capacity in Mbps for utilisation % (default 10.0)
    t_start      : reference epoch for timestamp_s column (default: now)

    Returns a handle dict to pass to stop_switch_monitor().

    Integration example in mininet_traffic_test.py
    -----------------------------------------------
        from collect_switch_metrics import start_switch_monitor, stop_switch_monitor

        # in main(), after net.start():
        t0      = time.time()
        monitor = start_switch_monitor(net.get('s1'), interval=1.0,
                                       link_bw_mbps=10.0, t_start=t0)
        run_test(net)          # your existing traffic test
        stop_switch_monitor(monitor)
    """
    name = switch if isinstance(switch, str) else switch.name
    if t_start is None:
        t_start = time.time()

    interfaces = get_switch_interfaces(name)
    if not interfaces:
        print("  Warning: no interfaces found on " + name)

    print("*** Switch monitor: {}  interfaces={}  interval={}s  capacity={} Mbps".format(
        name, interfaces, interval, link_bw_mbps))

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_poll_loop,
        args=(name, interfaces, interval, link_bw_mbps, stop_event, t_start),
        daemon=True,
    )
    thread.start()
    return {"thread": thread, "stop_event": stop_event, "interfaces": interfaces}


def stop_switch_monitor(handle):
    """Signal the polling thread to stop and wait for it to finish."""
    handle["stop_event"].set()
    handle["thread"].join(timeout=5)
    print("*** Switch monitor stopped. CSVs in ./" + RESULTS_DIR + "/")


# ---------------------------------------------------------------------------
# 7.  STANDALONE ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Poll tc qdisc + OVS flow stats from a Mininet switch."
    )
    parser.add_argument("--switch",   default="s1")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--bw",       type=float, default=10.0)
    args = parser.parse_args()

    if os.getuid() != 0:
        print("ERROR: requires root. Run with sudo.")
        raise SystemExit(1)

    t0     = time.time()
    handle = start_switch_monitor(args.switch, args.interval, args.bw, t0)
    try:
        print("Collecting for {}s -- Ctrl+C to stop early.".format(int(args.duration)))
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_switch_monitor(handle)
