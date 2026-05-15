#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  EMMDS — one-command launcher
#  Usage:  ./start.sh          (UI + API, default)
#          ./start.sh --ui     (Streamlit only)
#          ./start.sh --api    (FastAPI only)
#          ./start.sh --test   (Iris demo, no servers)
# ─────────────────────────────────────────────────────────────
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$DIR/venv/bin/python"
PID_API=""

# ── colour helpers ────────────────────────────────────────────
bold='\033[1m'; reset='\033[0m'
green='\033[0;32m'; cyan='\033[0;36m'; red='\033[0;31m'; yellow='\033[0;33m'

banner() {
  echo ""
  echo -e "${cyan}${bold}╔══════════════════════════════════════════════════════╗${reset}"
  echo -e "${cyan}${bold}║      EMMDS — Explainable AI Decision System          ║${reset}"
  echo -e "${cyan}${bold}╚══════════════════════════════════════════════════════╝${reset}"
  echo ""
}

# ── sanity check ─────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
  echo -e "${red}Error: venv not found at $DIR/venv${reset}"
  echo "Run:  python3 -m venv venv && venv/bin/pip install -r requirements.txt"
  exit 1
fi

# ── cleanup on Ctrl+C / exit ─────────────────────────────────
cleanup() {
  echo ""
  echo -e "${yellow}Shutting down EMMDS...${reset}"
  [ -n "$PID_API" ] && kill "$PID_API" 2>/dev/null && echo "  API stopped."
  echo -e "${green}Done. Goodbye.${reset}"
  exit 0
}
trap cleanup INT TERM

MODE="${1:-}"

banner

# ── test / demo mode ─────────────────────────────────────────
if [ "$MODE" = "--test" ]; then
  echo -e "${bold}Running demo on Iris dataset...${reset}"
  cd "$DIR"
  "$PYTHON" run.py --test
  exit 0
fi

# ── API only ─────────────────────────────────────────────────
if [ "$MODE" = "--api" ]; then
  echo -e "${bold}Starting FastAPI backend...${reset}"
  echo -e "  ${green}API docs  →  http://localhost:8000/docs${reset}"
  echo ""
  cd "$DIR"
  "$PYTHON" run.py --api
  exit 0
fi

# ── UI only ──────────────────────────────────────────────────
if [ "$MODE" = "--ui" ]; then
  echo -e "${bold}Starting Streamlit UI...${reset}"
  echo -e "  ${green}UI  →  http://localhost:8501${reset}"
  echo ""
  cd "$DIR"
  "$PYTHON" run.py --ui
  exit 0
fi

# ── default: both UI + API ────────────────────────────────────
echo -e "  ${green}UI   →  http://localhost:8501${reset}"
echo -e "  ${green}API  →  http://localhost:8000/docs${reset}"
echo ""
echo -e "  Press ${bold}Ctrl+C${reset} to stop both."
echo ""

cd "$DIR"

# Start FastAPI in background
"$PYTHON" run.py --api &
PID_API=$!
echo -e "  ${cyan}API started (pid $PID_API)${reset}"

# Give API 2 seconds to bind, then open browser
(sleep 2 && open "http://localhost:8501" 2>/dev/null || true) &

# Start Streamlit in foreground (blocks until Ctrl+C)
"$PYTHON" run.py --ui
