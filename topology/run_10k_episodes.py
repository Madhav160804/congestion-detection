import os
import sys
import glob
import subprocess

EPISODES = 20
DURATION = 500

def main():
    print("="*60)
    print(f" SDN Congestion Detetion: Multi-Episode Orchestrator")
    print(f" Executing {EPISODES} episodes of {DURATION} seconds each.")
    print("="*60)

    for ep in range(EPISODES):
        print(f"\n[Episode {ep+1}/{EPISODES}] Cleaning Mininet kernel state...")
        subprocess.run("echo '123#Madhav' | sudo -S mn -c", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        seed = 42 + ep
        outdir = f"episode_{ep}"

        print(f"[Episode {ep+1}/{EPISODES}] Launching Mininet Physics simulation (Seed: {seed})...")
        print(f"   => sudo python3 topology/mytopo.py --duration {DURATION} --seed {seed} --outdir raw_data/{outdir}")
        
        # Execute Episode
        subprocess.run(
            f"echo '123#Madhav' | sudo -S python3 topology/mytopo.py --duration {DURATION} --seed {seed} --outdir raw_data/{outdir}",
            shell=True
        )

    print("\n[Orchestrator] All physical executions complete! Splicing CSVs into seamless timeline...")

    import pandas as pd
    os.makedirs("iperf_results", exist_ok=True)
    os.makedirs("switch_results", exist_ok=True)
    os.makedirs("host_results", exist_ok=True)

    # 1. Splice Iperf
    iperf_dfs = []
    ping_dfs = []
    qdisc_dfs = []
    flow_dfs = []
    host_dfs = []

    for ep in range(EPISODES):
        offset = ep * DURATION
        dir_path = f"raw_data/episode_{ep}"

        # Iperf
        ipath = os.path.join(dir_path, "iperf_timeseries.csv")
        if os.path.exists(ipath):
            df = pd.read_csv(ipath)
            df["absolute_time_s"] += offset
            df["interval_start_s"] += offset
            df["interval_end_s"] += offset
            df["scheduled_flow_start_s"] += offset
            iperf_dfs.append(df)

        # Ping
        ppath = os.path.join(dir_path, "ping_timeseries.csv")
        if os.path.exists(ppath):
            df = pd.read_csv(ppath)
            # handle summary rows safely
            df["unix_timestamp_s"] = pd.to_numeric(df["unix_timestamp_s"], errors='coerce')
            df.loc[df["unix_timestamp_s"].notna(), "unix_timestamp_s"] += offset
            ping_dfs.append(df)

        # Qdisc
        qpath = os.path.join(dir_path.replace("raw_data/episode_", "raw_data/switch_results/episode_"), "qdisc_metrics.csv")
        if os.path.exists(qpath):
            df = pd.read_csv(qpath)
            df["timestamp_s"] += offset
            qdisc_dfs.append(df)
            
        qpath2 = os.path.join(dir_path, "qdisc_metrics.csv")  # In case outdir was direct
        if os.path.exists(qpath2):
            df = pd.read_csv(qpath2)
            df["timestamp_s"] += offset
            qdisc_dfs.append(df)

        # Host
        hpath = os.path.join(dir_path, "tcp_socket_metrics.csv")
        if os.path.exists(hpath):
            df = pd.read_csv(hpath)
            df["timestamp_s"] += offset
            host_dfs.append(df)

    if iperf_dfs: pd.concat(iperf_dfs, ignore_index=True).to_csv("iperf_results/iperf_timeseries.csv", index=False)
    if ping_dfs: pd.concat(ping_dfs, ignore_index=True).to_csv("iperf_results/ping_timeseries.csv", index=False)
    if qdisc_dfs: pd.concat(qdisc_dfs, ignore_index=True).to_csv("switch_results/qdisc_metrics.csv", index=False)
    if host_dfs: pd.concat(host_dfs, ignore_index=True).to_csv("host_results/tcp_socket_metrics.csv", index=False)

    print("\n[Done] Seamless 10,000-second execution extracted into local folders!")

if __name__ == '__main__':
    main()
