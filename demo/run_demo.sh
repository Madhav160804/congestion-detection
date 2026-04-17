#!/bin/bash
# ============================================================
#  SDN CongestionShield — Demo Launcher
#  Run from the project root inside WSL2:
#      bash demo/run_demo.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

cd "$ROOT"

echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║      🛡  SDN CongestionShield Demo Launcher      ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Dependency checks ────────────────────────────────────────────────────
echo "[1/4] Checking dependencies…"

for pkg in python3 mn ovs-vsctl iperf3 ethtool curl; do
    if ! command -v "$pkg" &>/dev/null; then
        echo "  MISSING: $pkg"
        if [[ "$pkg" == "iperf3" ]]; then
            echo "  → Installing iperf3…"
            sudo apt-get install -y iperf3 -qq
        elif [[ "$pkg" == "ethtool" ]]; then
            echo "  → Installing ethtool…"
            sudo apt-get install -y ethtool -qq
        else
            echo "  ERROR: Please install $pkg and re-run."
            exit 1
        fi
    fi
done

python3 -c "import flask" 2>/dev/null || {
    echo "  Installing Flask…"
    pip3 install flask --quiet
}

echo "  All dependencies satisfied."

# ── 2. Check model exists ───────────────────────────────────────────────────
echo ""
echo "[2/4] Checking model artefacts…"

if [[ ! -f "$ROOT/model/best_model.pkl" ]]; then
    echo "  ERROR: model/best_model.pkl not found. Run train_model.py first."
    exit 1
fi
if [[ ! -f "$ROOT/model/feature_names.txt" ]]; then
    echo "  ERROR: model/feature_names.txt not found."
    exit 1
fi
echo "  Model: OK ($(wc -l < "$ROOT/model/feature_names.txt") features)"

# ── 3. Clean up any stale Mininet state ────────────────────────────────────
echo ""
echo "[3/4] Cleaning stale Mininet state…"
sudo mn -c &>/dev/null || true
sudo pkill -9 -f "iperf3" 2>/dev/null || true
echo "  Clean."

# ── 4. Launch inference server ─────────────────────────────────────────────
echo ""
echo "[4/4] Starting CongestionShield inference server…"
echo ""
echo "  ┌─────────────────────────────────────────────────┐"
echo "  │  Dashboard : http://localhost:5050               │"
echo "  │  API state : http://localhost:5050/api/state     │"
echo "  │                                                 │"
echo "  │  Click  ⚡ Inject Traffic Surge  to demo        │"
echo "  │  Click  ✓ Clear Traffic          to recover     │"
echo "  │                                                 │"
echo "  │  Press  Ctrl+C  to stop everything              │"
echo "  └─────────────────────────────────────────────────┘"
echo ""

# Open browser on the Windows side (works from WSL2)
sleep 3 && cmd.exe /c "start http://localhost:5050" 2>/dev/null &

# Run the server (needs root for Mininet + tc)
sudo python3 "$SCRIPT_DIR/inference_server.py"
