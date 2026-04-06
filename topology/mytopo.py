#!/usr/bin/env python3
"""
Mininet Traffic Congestion Test
================================
Topology : h1 --10Mbps-- s1 --10Mbps-- h2
Test     : Gradually ramp up iperf TCP flows + high-frequency pings over ~3 min.
           New flows stop at t=120 s; everything finishes by t=180 s.

Output files (all under ./iperf_results/):
  flow_N.json              – raw iperf3 JSON per flow
  ping_N.log               – raw ping output per stream
  iperf_timeseries.csv     – one row per second per flow (iperf metrics)
  ping_timeseries.csv      – one row per ping reply (latency / loss metrics)
"""

import os
import re
import sys
import time
import csv
import json
import signal
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, Controller
from mininet.link import TCLink
from mininet.log import setLogLevel, info


import random
import math

from collect_switch_metrics import start_switch_monitor, stop_switch_monitor
from collect_host_metrics   import start_host_monitor,   stop_host_monitor

def generate_flow_schedule(
    stop_arrivals_at: float = 120.0,
    mean_interarrival: float = 10.0,
    bw_min: float = 10.0,
    bw_max: float = 20.0,
    dur_min: float = 30.0,
    dur_max: float = 100.0,
    seed: int = None,
):
    """
    Generate a random flow schedule as a list of
        (start_time_s, bandwidth_mbps, duration_s)
    tuples, sorted by start time.

    Arrival times follow a Poisson process: inter-arrival gaps are drawn
    from an Exponential distribution with mean = mean_interarrival seconds.

    Parameters
    ----------
    stop_arrivals_at   : no new flows are scheduled to start after this time
    mean_interarrival  : average gap between consecutive flow arrivals (seconds)
                         λ = 1 / mean_interarrival  (flows per second)
    bw_min / bw_max    : uniform bandwidth range (Mbps)
    dur_min / dur_max  : uniform duration range (seconds)
    seed               : optional RNG seed for reproducibility
    """
    rng = random.Random(seed)

    schedule = []
    current_time = 0.0

    while True:
        # Inter-arrival time ~ Exponential(λ = 1 / mean_interarrival)
        # Inverse-CDF method: X = -mean * ln(U),  U ~ Uniform(0, 1)
        interarrival = -mean_interarrival * math.log(rng.random())
        current_time += interarrival

        if current_time > stop_arrivals_at:
            break

        bw  = round(rng.uniform(bw_min, bw_max), 2)
        dur = round(rng.uniform(dur_min, dur_max), 1)

        schedule.append((round(current_time, 2), bw, dur))

    return schedule


# ── Output directory ─────────────────────────────────────────────────────────
RESULTS_DIR = "iperf_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# 1.  TOPOLOGY
# ────────────────────────────────────────────────────────────────────────────
def build_topology():
    """Return a started Mininet: h1 -- s1 -- h2, both links at 10 Mbps."""
    net = Mininet(
        switch=OVSKernelSwitch,
        controller=Controller,
        link=TCLink,
        autoSetMacs=True,
    )

    info("*** Adding controller\n")
    net.addController("c0")

    info("*** Adding hosts\n")
    h1 = net.addHost("h1", ip="10.0.0.1/24")
    h2 = net.addHost("h2", ip="10.0.0.2/24")

    info("*** Adding switch\n")
    s1 = net.addSwitch("s1")

    info("*** Adding links (bw=10 Mbps)\n")
    # max_queue_size limits the qdisc buffer to ~20 packets.
    # Without this the default queue is hundreds of packets deep
    # (bufferbloat), so tc delays rather than drops — TCP never sees
    # a loss and retransmits stay at 0. A 20-packet limit at 10 Mbps
    # gives ~24 ms of buffering, tight enough to force real drops
    # and retransmissions when competing flows saturate the link.
    net.addLink(h1, s1, delay="30ms", bw=25, max_queue_size=30)
    net.addLink(s1, h2, delay="30ms", bw=25, max_queue_size=30)

    #h1.cmd("sysctl -w net.ipv4.tcp_congestion_control=reno")
    #h2.cmd("sysctl -w net.ipv4.tcp_congestion_control=reno")

    #[ h1.cmd(f"ethtool -K {inf} tso off gso off") for inf in h1.intfNames() ]
    #[ h2.cmd(f"ethtool -K {inf} tso off gso off") for inf in h2.intfNames() ]
    #[ s1.cmd(f"ethrool -K {inf} tso off gso off") for inf in s1.intfNames() ]
    info("*** Starting network\n")
    net.start()
    return net


# ────────────────────────────────────────────────────────────────────────────
# 2.  SCHEDULES
# ────────────────────────────────────────────────────────────────────────────
# (start_time_s, target_bw_Mbps, duration_s)
# Flows are launched aggressively so that by t=30s the combined target
# bandwidth already exceeds the 10 Mbps link, guaranteeing real queue
# overflow, packet drops, and TCP retransmissions.
FLOW_SCHEDULE = generate_flow_schedule()

# (start_time_s, stop_time_s, ping_interval_s)
PING_SCHEDULE = [
    (0,   60,  1.0),   # 1 ping/s
    (20,  120, 0.5),   # 2 pings/s
    (60,  180, 0.2),   # 5 pings/s
]


# ────────────────────────────────────────────────────────────────────────────
# 3.  TRAFFIC TEST
# ────────────────────────────────────────────────────────────────────────────
def run_test(net):
    h1, h2 = net.get("h1"), net.get("h2")

    # iperf3 only handles one client at a time, so each flow gets its own
    # dedicated server instance listening on a unique port (BASE_PORT + flow_id).
    BASE_PORT = 5201
    info("*** Pre-starting one iperf3 server per flow on h2\n")
    for fid in range(len(FLOW_SCHEDULE)):
        port = BASE_PORT + fid
        h2.cmd(f"iperf3 -s -p {port} -D --pidfile /tmp/iperf3_{port}.pid")
    time.sleep(1)   # give all servers time to bind

    flow_procs    = []   # list of (flow_id, Popen)
    ping_procs    = []   # list of Popen (with ._tag attribute)
    t_start       = time.time()
    next_flow_idx = 0

    def elapsed():
        return time.time() - t_start

    info("*** Test started – ramping traffic for 120 s, coasting to 180 s\n")

    while elapsed() < 200:
        t = elapsed()

        # ── launch iperf flows on schedule ────────────────────────────────
        while (next_flow_idx < len(FLOW_SCHEDULE) and
               t >= FLOW_SCHEDULE[next_flow_idx][0]):
            start_t, bw, dur = FLOW_SCHEDULE[next_flow_idx]
            fid      = next_flow_idx
            port     = BASE_PORT + fid
            log_file = os.path.join(RESULTS_DIR, f"flow_{fid}.json")
            cmd = (
                f"iperf3 -c {h2.IP()} "
                f"-p {port} "     # dedicated server port for this flow
                f"-b {bw}M "
                f"-t {dur} "
                f"-i 1 "          # 1-second interval reporting
                f"--json "
                f"> {log_file} 2>&1"
            )
            info(f"  [t={t:.1f}s] Launching iperf flow {fid} "
                 f"(port {port}): {bw} Mbps for {dur}s\n")
            proc = h1.popen(cmd, shell=True)
            flow_procs.append((fid, proc))
            next_flow_idx += 1

        # ── launch ping streams on schedule ──────────────────────────────
        for sidx, (ps, pe, interval) in enumerate(PING_SCHEDULE):
            tag = f"__ping_{sidx}__"
            if t >= ps and t < pe:
                if not any(getattr(p, "_tag", "") == tag for p in ping_procs):
                    info(f"  [t={t:.1f}s] Starting ping stream {sidx} "
                         f"(interval={interval}s)\n")
                    # -D prefixes each line with a Unix timestamp
                    pcmd = (
                        f"ping -i {interval} -D {h2.IP()} "
                        f"> {RESULTS_DIR}/ping_{sidx}.log 2>&1"
                    )
                    p = h1.popen(pcmd, shell=True)
                    p._tag = tag
                    ping_procs.append(p)

        # ── stop ping streams whose window has closed ─────────────────────
        for sidx, (ps, pe, interval) in enumerate(PING_SCHEDULE):
            tag = f"__ping_{sidx}__"
            if t >= pe:
                for p in list(ping_procs):
                    if getattr(p, "_tag", "") == tag:
                        try:
                            p.send_signal(signal.SIGINT)
                        except Exception:
                            pass
                        ping_procs.remove(p)

        time.sleep(0.5)

    # ── cleanup ───────────────────────────────────────────────────────────
    info("*** Stopping remaining flows\n")
    for p in list(ping_procs):
        try:
            p.send_signal(signal.SIGINT)
        except Exception:
            pass

    for fid, p in flow_procs:
        p.wait()

    # kill each server by its pidfile to avoid disrupting other processes
    for fid in range(len(FLOW_SCHEDULE)):
        port = BASE_PORT + fid
        h2.cmd(f"kill $(cat /tmp/iperf3_{port}.pid 2>/dev/null) 2>/dev/null || true")
    info("*** Test complete\n")


# ────────────────────────────────────────────────────────────────────────────
# 4.  COMPILE IPERF CSV
# ────────────────────────────────────────────────────────────────────────────
def compile_iperf_csv():
    """
    One row per 1-second interval per flow → iperf_timeseries.csv

    Columns
    -------
    Identification
      flow_id                      index into FLOW_SCHEDULE (0-based)
      target_bw_mbps               requested bandwidth for this flow
      scheduled_flow_start_s       when in the test this flow was launched
      interval_index               0-based second within the flow

    Timing
      absolute_time_s              scheduled_start + interval midpoint
      interval_start_s             seconds from flow start (left edge)
      interval_end_s               seconds from flow start (right edge)

    Throughput
      bytes_transferred            bytes sent in this 1-s window
      bits_per_second              raw bps
      mbps_achieved                bps / 1e6  (4 d.p.)

    Reliability / Congestion
      retransmits                  TCP segments retransmitted this second
      snd_cwnd_bytes               TCP congestion-window size at interval end
      pmtu                         path MTU discovered by the sender

    Latency  (TCP-layer, sender-measured)
      rtt_us                       smoothed RTT in microseconds
      rttvar_us                    RTT variance in microseconds

    Omit flag
      omitted                      True if iperf3 treated this as a warm-up
                                   interval (--omit mode)

    Per-flow summary  (repeated on every row for easy filtering / pivoting)
      cpu_utilization_local_pct    sender CPU % for the whole flow
      cpu_utilization_remote_pct   receiver CPU % for the whole flow
      congestion_algorithm         TCP CC in use  (e.g. cubic, bbr)
      max_snd_cwnd_bytes           peak congestion window over the whole flow
    """
    rows = []

    for fid, (start_t, bw, dur) in enumerate(FLOW_SCHEDULE):
        log_path = os.path.join(RESULTS_DIR, f"flow_{fid}.json")
        if not os.path.exists(log_path):
            info(f"  Warning: {log_path} not found – skipping\n")
            continue

        with open(log_path) as f:
            raw = f.read().strip()

        json_start = raw.find("{")
        if json_start == -1:
            info(f"  Warning: no JSON in {log_path} – skipping\n")
            continue

        try:
            data = json.loads(raw[json_start:])
        except json.JSONDecodeError as e:
            info(f"  Warning: JSON parse error in {log_path}: {e}\n")
            continue

        intervals = data.get("intervals", [])
        if not intervals:
            info(f"  Warning: no interval data in {log_path} – skipping\n")
            continue

        # ── flow-level summary from data["end"] ───────────────────────────
        end        = data.get("end", {})
        cpu        = end.get("cpu_utilization_percent", {})
        cpu_local  = cpu.get("host_total",   "")
        cpu_remote = cpu.get("remote_total", "")
        cc_algo    = end.get("sender_tcp_congestion", "")

        max_cwnd = ""
        end_streams = end.get("streams", [])
        if end_streams:
            max_cwnd = end_streams[0].get("sender", {}).get("max_snd_cwnd", "")

        # ── per-interval rows ─────────────────────────────────────────────
        for idx, interval in enumerate(intervals):
            s = interval.get("sum", {})

            i_start  = s.get("start", idx)
            i_end    = s.get("end",   idx + 1)
            midpoint = (i_start + i_end) / 2.0

            # retransmits, RTT, cwnd, pmtu all live in
            # interval["streams"][0] (TCP sender stream block).
            # interval["sum"] does NOT carry retransmits — it is always 0 there.
            stream0     = (interval.get("streams") or [{}])[0]
            retransmits = stream0.get("retransmits", "")
            rtt_us      = stream0.get("rtt",         "")
            rttvar      = stream0.get("rttvar",      "")
            cwnd        = stream0.get("snd_cwnd",    "")
            pmtu        = stream0.get("pmtu",        "")

            rows.append({
                # identification
                "flow_id":                     fid,
                "target_bw_mbps":              bw,
                "scheduled_flow_start_s":      start_t,
                "interval_index":              idx,
                # timing
                "absolute_time_s":             round(start_t + midpoint, 3),
                "interval_start_s":            round(i_start, 3),
                "interval_end_s":              round(i_end,   3),
                # throughput
                "bytes_transferred":           s.get("bytes", ""),
                "bits_per_second":             round(s.get("bits_per_second", 0)),
                "mbps_achieved":               round(s.get("bits_per_second", 0) / 1e6, 4),
                # reliability / congestion
                "retransmits":                 retransmits,
                "snd_cwnd_bytes":              cwnd,
                "pmtu":                        pmtu,
                # latency
                "rtt_us":                      rtt_us,
                "rttvar_us":                   rttvar,
                # omit flag
                "omitted":                     s.get("omitted", False),
                # flow-level summary (repeated)
                "cpu_utilization_local_pct":   cpu_local,
                "cpu_utilization_remote_pct":  cpu_remote,
                "congestion_algorithm":        cc_algo,
                "max_snd_cwnd_bytes":          max_cwnd,
            })

        info(f"  iperf flow {fid}: {len(intervals)} interval rows\n")

    csv_path = os.path.join(RESULTS_DIR, "iperf_timeseries.csv")
    if rows:
        rows.sort(key=lambda r: (r["absolute_time_s"], r["flow_id"]))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        info(f"*** iperf CSV: {csv_path}  ({len(rows)} rows)\n")
    else:
        info("*** No iperf interval data – CSV not written\n")

    return csv_path


# ────────────────────────────────────────────────────────────────────────────
# 5.  COMPILE PING CSV
# ────────────────────────────────────────────────────────────────────────────
# Regex for lines produced by `ping -D`
#
#  Reply:   [1700000000.123] 64 bytes from 10.0.0.2: icmp_seq=1 ttl=64 time=0.456 ms
#  Timeout: [1700000000.789] no answer yet for icmp_seq=5
#  Summary: 5 packets transmitted, 4 received, 20% packet loss, time 4003ms
#  RTT:     rtt min/avg/max/mdev = 0.123/0.456/0.789/0.111 ms
#
_REPLY_RE   = re.compile(
    r'^\[(\d+\.\d+)\].*icmp_seq=(\d+)\s+ttl=(\d+)\s+time=([\d.]+)\s+ms'
)
_TIMEOUT_RE = re.compile(
    r'^\[(\d+\.\d+)\].*no answer yet for icmp_seq=(\d+)'
)
_SUMMARY_RE = re.compile(
    r'(\d+) packets transmitted,\s*(\d+) received,\s*([\d.]+)% packet loss'
)
_RTTSTAT_RE = re.compile(
    r'rtt min/avg/max/mdev\s*=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s+ms'
)


def compile_ping_csv():
    """
    Parse every ping_N.log → ping_timeseries.csv

    Columns
    -------
    Per-reply rows  (status = 'reply' or 'timeout')
      stream_id           index into PING_SCHEDULE (0-based)
      ping_interval_s     the -i value used for this stream
      icmp_seq            ICMP sequence number
      unix_timestamp_s    wall-clock time from the -D flag
      status              'reply' | 'timeout'
      rtt_ms              round-trip time in ms  (blank for timeouts)
      ttl                 IP TTL of the reply     (blank for timeouts)
      pkts_transmitted    blank (populated on summary row only)
      pkts_received       blank
      pkt_loss_pct        blank
      rtt_min_ms          blank
      rtt_avg_ms          blank
      rtt_max_ms          blank
      rtt_mdev_ms         blank

    Per-stream summary row  (status = 'summary', icmp_seq = 'SUMMARY')
      stream_id, ping_interval_s, icmp_seq='SUMMARY',
      unix_timestamp_s='', status='summary', rtt_ms='', ttl='',
      pkts_transmitted, pkts_received, pkt_loss_pct,
      rtt_min_ms, rtt_avg_ms, rtt_max_ms, rtt_mdev_ms
    """
    all_rows     = []
    summary_rows = []

    for sidx, (ps, pe, interval) in enumerate(PING_SCHEDULE):
        log_path = os.path.join(RESULTS_DIR, f"ping_{sidx}.log")
        if not os.path.exists(log_path):
            info(f"  Warning: {log_path} not found – skipping\n")
            continue

        with open(log_path) as f:
            lines = f.readlines()

        reply_count = 0
        for line in lines:
            line = line.strip()

            m = _REPLY_RE.search(line)
            if m:
                all_rows.append({
                    "stream_id":        sidx,
                    "ping_interval_s":  interval,
                    "icmp_seq":         int(m.group(2)),
                    "unix_timestamp_s": float(m.group(1)),
                    "status":           "reply",
                    "rtt_ms":           float(m.group(4)),
                    "ttl":              int(m.group(3)),
                    "pkts_transmitted": "",
                    "pkts_received":    "",
                    "pkt_loss_pct":     "",
                    "rtt_min_ms":       "",
                    "rtt_avg_ms":       "",
                    "rtt_max_ms":       "",
                    "rtt_mdev_ms":      "",
                })
                reply_count += 1
                continue

            m = _TIMEOUT_RE.search(line)
            if m:
                all_rows.append({
                    "stream_id":        sidx,
                    "ping_interval_s":  interval,
                    "icmp_seq":         int(m.group(2)),
                    "unix_timestamp_s": float(m.group(1)),
                    "status":           "timeout",
                    "rtt_ms":           "",
                    "ttl":              "",
                    "pkts_transmitted": "",
                    "pkts_received":    "",
                    "pkt_loss_pct":     "",
                    "rtt_min_ms":       "",
                    "rtt_avg_ms":       "",
                    "rtt_max_ms":       "",
                    "rtt_mdev_ms":      "",
                })

        # ── summary statistics from the last few lines ────────────────────
        tail = " ".join(lines[-5:])
        sm   = _SUMMARY_RE.search(tail)
        rtm  = _RTTSTAT_RE.search(tail)

        summary_rows.append({
            "stream_id":        sidx,
            "ping_interval_s":  interval,
            "icmp_seq":         "SUMMARY",
            "unix_timestamp_s": "",
            "status":           "summary",
            "rtt_ms":           "",
            "ttl":              "",
            "pkts_transmitted": sm.group(1)  if sm  else "",
            "pkts_received":    sm.group(2)  if sm  else "",
            "pkt_loss_pct":     sm.group(3)  if sm  else "",
            "rtt_min_ms":       rtm.group(1) if rtm else "",
            "rtt_avg_ms":       rtm.group(2) if rtm else "",
            "rtt_max_ms":       rtm.group(3) if rtm else "",
            "rtt_mdev_ms":      rtm.group(4) if rtm else "",
        })

        info(f"  ping stream {sidx}: {reply_count} reply rows parsed\n")

    all_rows.sort(key=lambda r: (
        r["unix_timestamp_s"] if r["unix_timestamp_s"] != "" else float("inf")
    ))
    combined = all_rows + summary_rows

    csv_path = os.path.join(RESULTS_DIR, "ping_timeseries.csv")
    if combined:
        fieldnames = [
            "stream_id", "ping_interval_s", "icmp_seq",
            "unix_timestamp_s", "status", "rtt_ms", "ttl",
            "pkts_transmitted", "pkts_received", "pkt_loss_pct",
            "rtt_min_ms", "rtt_avg_ms", "rtt_max_ms", "rtt_mdev_ms",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined)
        info(f"*** ping CSV: {csv_path}  ({len(all_rows)} reply rows + "
             f"{len(summary_rows)} summary rows)\n")
    else:
        info("*** No ping data – CSV not written\n")

    return csv_path


# ────────────────────────────────────────────────────────────────────────────
# 6.  ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────
def main():
    setLogLevel("info")

    net = build_topology()
    try:
        info("*** Verifying connectivity\n")
        net.pingAll()
        monitor      = start_switch_monitor(net.get('s1'), interval=1.0, link_bw_mbps=25.0)
        host_monitor = start_host_monitor(net.get('h1'),  interval=1.0)
        run_test(net)
        stop_switch_monitor(monitor)
        stop_host_monitor(host_monitor)
    finally:
        info("*** Stopping network\n")
        net.stop()

    info("*** Compiling CSVs\n")
    compile_iperf_csv()
    compile_ping_csv()
    info("*** Done.  Results in ./iperf_results/\n")


if __name__ == "__main__":
    if os.getuid() != 0:
        print("ERROR: Mininet requires root.  "
              "Run with: sudo python3 mininet_traffic_test.py")
        sys.exit(1)
    main()
