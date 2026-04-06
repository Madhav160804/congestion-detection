#!/usr/bin/env python3
"""
Host-Level TCP Socket Metrics Collector
========================================
Polls `ss -tin state established` on a Mininet host node every second to
capture per-socket TCP kernel state: smoothed RTT, minimum RTT, congestion
window, slow-start threshold, retransmit counts, and delivery rate.

Why `ss -tin` over relying solely on iperf RTT?
-------------------------------------------------
  iperf3 samples RTT at 1-second intervals and reports an average over the
  interval. `ss` reads the kernel tcp_info struct directly and gives:

    rtt       : kernel SRTT (smoothed RTT) — updated every ACK, more
                accurate than iperf's once-per-second sample.

    min_rtt   : minimum RTT observed since the socket was created.
                Because the minimum is taken before the queue builds up,
                min_rtt ≈ pure propagation delay (2 × link_delay × n_links).
                This is the gold-standard baseline for the Congestion
                Evidence Framework: it is data-derived, not assumed.

    retrans   : cumulative TCP retransmit count — independent cross-check
                against iperf's retransmit reporting.

    cwnd      : current congestion window (segments).  A sudden halving of
                cwnd confirms a loss event occurred (TCP Reno/CUBIC behaviour).

    ssthresh  : slow-start threshold.  When cwnd > ssthresh the socket is in
                congestion-avoidance mode.  A drop in ssthresh is TCP's own
                acknowledgement of congestion.

Usage (imported, as called from mytopo.py)
------------------------------------------
    from collect_host_metrics import start_host_monitor, stop_host_monitor
    monitor = start_host_monitor(net.get('h1'), interval=1.0, t_start=t0)
    run_test(net)
    stop_host_monitor(monitor)

Output
------
    host_results/tcp_socket_metrics.csv
"""

import re
import os
import csv
import time
import threading

RESULTS_DIR = "host_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

HOST_CSV = os.path.join(RESULTS_DIR, "tcp_socket_metrics.csv")

# ── ss output field parsers ─────────────────────────────────────────────────
_RTT_RE      = re.compile(r'\brtt:([\d.]+)/([\d.]+)')          # rtt:srtt/rttvar
_CWND_RE     = re.compile(r'\bcwnd:(\d+)')
_SSTHRESH_RE = re.compile(r'\bssthresh:(\d+)')
_RETRANS_RE  = re.compile(r'\bretrans:(\d+)/(\d+)')             # since_recv/total
_MINRTT_RE   = re.compile(r'\bminrtt:([\d.]+)')
_BYTES_RE    = re.compile(r'\bbytes_sent:(\d+)')
_DELIV_RE    = re.compile(r'delivery_rate\s+([\d.]+)(Kbps|Mbps|Gbps|bps)')
_LOST_RE     = re.compile(r'\blost:(\d+)')
_CC_RE       = re.compile(r'^\s+(cubic|bbr|reno|vegas|htcp)\b')

HOST_CSV_FIELDS = [
    "timestamp_s",
    "socket_count",
    "rtt_ms_mean",        "rtt_ms_max",
    "rttvar_ms_mean",
    "minrtt_ms_mean",     "minrtt_ms_min",
    "cwnd_segs_mean",     "cwnd_segs_min",
    "ssthresh_mean",
    "retrans_total_sum",  "retrans_delta_sum",
    "lost_total_sum",     "lost_delta_sum",
    "bytes_sent_sum",     "bytes_sent_delta_sum",
    "delivery_mbps_mean",
]


def _parse_ss_output(raw):
    """
    Parse one complete `ss -tin state established` dump.

    Returns a list of dicts, one per ESTABLISHED socket found.
    `ss -tin` prints two lines per socket:
      Line 1 (header): State RecvQ SendQ LocalAddr:Port PeerAddr:Port
      Line 2 (info  ): tcp-cc-name key:value key:value ...
    We scan for the info line by looking for known TCP fields.
    """
    sockets = []
    lines   = raw.splitlines()
    i       = 0
    while i < len(lines):
        line = lines[i]
        # Detect socket header line
        if "ESTAB" in line or "ESTABLISHED" in line:
            # The info block follows — may span multiple lines (joined by whitespace)
            info = ""
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or
                                       lines[j][0] in (" ", "\t")):
                info += " " + lines[j]
                j += 1
            i = j

            sock = {}

            m = _RTT_RE.search(info)
            if m:
                sock["rtt_ms"]    = float(m.group(1))
                sock["rttvar_ms"] = float(m.group(2))

            m = _MINRTT_RE.search(info)
            if m:
                sock["minrtt_ms"] = float(m.group(1))

            m = _CWND_RE.search(info)
            if m:
                sock["cwnd_segs"] = int(m.group(1))

            m = _SSTHRESH_RE.search(info)
            if m:
                sock["ssthresh"] = int(m.group(1))

            m = _RETRANS_RE.search(info)
            if m:
                sock["retrans_total"] = int(m.group(2))

            m = _LOST_RE.search(info)
            if m:
                sock["lost_total"] = int(m.group(1))

            m = _BYTES_RE.search(info)
            if m:
                sock["bytes_sent"] = int(m.group(1))

            m = _DELIV_RE.search(info)
            if m:
                val  = float(m.group(1))
                unit = m.group(2)
                mult = {"bps": 1e-6, "Kbps": 1e-3, "Mbps": 1.0, "Gbps": 1e3}
                sock["delivery_mbps"] = val * mult.get(unit, 1.0)

            if sock:
                sockets.append(sock)
        else:
            i += 1

    return sockets


def _aggregate_sockets(sockets, prev_totals):
    """
    Aggregate per-socket stats into a single row.
    Compute deltas for cumulative counters (retrans, lost, bytes_sent).
    """
    if not sockets:
        return None, prev_totals

    def _mean(key):
        vals = [s[key] for s in sockets if key in s]
        return sum(vals) / len(vals) if vals else None

    def _min(key):
        vals = [s[key] for s in sockets if key in s]
        return min(vals) if vals else None

    def _max(key):
        vals = [s[key] for s in sockets if key in s]
        return max(vals) if vals else None

    def _sum(key):
        return sum(s.get(key, 0) for s in sockets)

    curr_retrans = _sum("retrans_total")
    curr_lost    = _sum("lost_total")
    curr_bytes   = _sum("bytes_sent")

    retrans_delta = max(0, curr_retrans - prev_totals.get("retrans", curr_retrans))
    lost_delta    = max(0, curr_lost    - prev_totals.get("lost",    curr_lost))
    bytes_delta   = max(0, curr_bytes   - prev_totals.get("bytes",   curr_bytes))

    new_prev = {"retrans": curr_retrans, "lost": curr_lost, "bytes": curr_bytes}

    row = {
        "socket_count":        len(sockets),
        "rtt_ms_mean":         _mean("rtt_ms"),
        "rtt_ms_max":          _max("rtt_ms"),
        "rttvar_ms_mean":      _mean("rttvar_ms"),
        "minrtt_ms_mean":      _mean("minrtt_ms"),
        "minrtt_ms_min":       _min("minrtt_ms"),
        "cwnd_segs_mean":      _mean("cwnd_segs"),
        "cwnd_segs_min":       _min("cwnd_segs"),
        "ssthresh_mean":       _mean("ssthresh"),
        "retrans_total_sum":   curr_retrans,
        "retrans_delta_sum":   retrans_delta,
        "lost_total_sum":      curr_lost,
        "lost_delta_sum":      lost_delta,
        "bytes_sent_sum":      curr_bytes,
        "bytes_sent_delta_sum": bytes_delta,
        "delivery_mbps_mean":  _mean("delivery_mbps"),
    }
    return row, new_prev


def _poll_loop(host, interval, stop_event, t_start):
    with open(HOST_CSV, "w", newline="") as f:
        writer     = csv.DictWriter(f, fieldnames=HOST_CSV_FIELDS)
        writer.writeheader()
        prev_time  = None
        prev_tot   = {}

        while not stop_event.is_set():
            poll_start = time.time()
            ts         = poll_start - t_start

            try:
                raw = host.cmd("ss -tin state established")
                sockets = _parse_ss_output(raw)
                row, prev_tot = _aggregate_sockets(sockets, prev_tot)
                if row is not None:
                    row["timestamp_s"] = round(ts, 3)
                    # round floats
                    for k, v in row.items():
                        if isinstance(v, float):
                            row[k] = round(v, 4)
                    writer.writerow(row)
                    f.flush()
            except Exception as e:
                pass   # transient errors during network teardown are expected

            prev_time = poll_start
            stop_event.wait(timeout=max(0.0, interval - (time.time() - poll_start)))


# ── Public API ───────────────────────────────────────────────────────────────

def start_host_monitor(host, interval=1.0, t_start=None):
    """
    Start a background thread polling `ss -tin` on the given Mininet host.

    Parameters
    ----------
    host     : Mininet host object (must support .cmd())
    interval : polling interval in seconds
    t_start  : reference epoch for timestamp_s  (default: now)

    Returns a handle dict to pass to stop_host_monitor().
    """
    if t_start is None:
        t_start = time.time()

    print(f"*** Host monitor: {host.name}  interval={interval}s  "
          f"→ {HOST_CSV}")

    stop_event = threading.Event()
    thread     = threading.Thread(
        target=_poll_loop,
        args=(host, interval, stop_event, t_start),
        daemon=True,
    )
    thread.start()
    return {"thread": thread, "stop_event": stop_event}


def stop_host_monitor(handle):
    """Signal the polling thread to stop and wait for it."""
    handle["stop_event"].set()
    handle["thread"].join(timeout=5)
    print(f"*** Host monitor stopped. CSV → {HOST_CSV}")
