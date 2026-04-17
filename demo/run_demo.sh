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
echo "[1/5] Checking dependencies…"

for pkg in python3 mn ovs-vsctl iperf3 ethtool; do
    if ! command -v "$pkg" &>/dev/null; then
        echo "  MISSING: $pkg"
        if [[ "$pkg" == "iperf3" ]]; then
            sudo apt-get install -y iperf3 -qq
        elif [[ "$pkg" == "ethtool" ]]; then
            sudo apt-get install -y ethtool -qq
        else
            echo "  ERROR: Please install $pkg and re-run."
            exit 1
        fi
    fi
done

sudo python3 -c "import flask, sklearn, pandas, numpy" 2>/dev/null || {
    echo "  Installing required Python libraries globally (requires root)..."
    sudo pip3 install flask scikit-learn pandas numpy --quiet --break-system-packages --ignore-installed
}

echo "  All dependencies satisfied."

# ── 2. Check base model artefacts ───────────────────────────────────────────
echo ""
echo "[2/5] Checking model artefacts…"

if [[ ! -f "$ROOT/model/best_model.pkl" ]]; then
    echo "  ERROR: model/best_model.pkl not found. Run models/train_model.py first."
    exit 1
fi
echo "  Base model: OK"

# ── 3. Train demo model (if not already done) ───────────────────────────────
echo ""
echo "[3/5] Preparing demo inference model…"

if [[ ! -f "$ROOT/model/demo_model.pkl" ]]; then
    echo "  Building demo_model.pkl (one-time, ~30s)…"
    python3 "$SCRIPT_DIR/train_demo_model.py"
else
    echo "  demo_model.pkl exists — skipping retrain."
fi

# ── 4. Clean up stale Mininet state ─────────────────────────────────────────
echo ""
echo "[4/5] Cleaning stale Mininet state…"
sudo mn -c &>/dev/null || true
sudo pkill -9 -f "iperf3" 2>/dev/null || true
echo "  Clean."

# ── 5. Launch inference server ───────────────────────────────────────────────
echo ""
echo "[5/5] Starting CongestionShield inference server…"
echo ""
echo "  ┌─────────────────────────────────────────────────┐"
echo "  │  Dashboard : http://localhost:5050               │"
echo "  │  API state : http://localhost:5050/api/state     │"
echo "  │                                                 │"
echo "  │  🎬 Video player: inside dashboard (simulated)  │"
echo "  │  Click  ⚡ Inject Traffic Surge  to demo        │"
echo "  │  Click  ✓ Clear Traffic          to recover     │"
echo "  │                                                 │"
echo "  │  Press  Ctrl+C  to stop everything              │"
echo "  └─────────────────────────────────────────────────┘"
echo ""

# Open browser on Windows side (works from WSL2)
sleep 3 && cmd.exe /c "start http://localhost:5050" 2>/dev/null &

# Run the server (needs root for Mininet + tc)
sudo python3 "$SCRIPT_DIR/inference_server.py"
