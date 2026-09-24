#!/usr/bin/env bash
# Headless background execution script for squirrel_detection_rfdetr_medium.ipynb
# Allows training to continue uninterrupted even when the IDE or SSH session is closed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p output
LOG_FILE="output/notebook_headless.log"
PID_FILE="output/train.pid"

# Check if a run is already active
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Error: Training process is already running with PID $OLD_PID."
        echo "Monitor logs with: tail -f $LOG_FILE"
        exit 1
    fi
fi

echo "=========================================================="
echo " Starting Headless Notebook Execution in Background"
echo "=========================================================="
echo "Notebook: squirrel_detection_rfdetr_medium.ipynb"
echo "Log file: $LOG_FILE"
echo ""

nohup uv run jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    squirrel_detection_rfdetr_medium.ipynb \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "Started successfully with PID: $PID"
echo ""
echo "You can now safely close your IDE or disconnect your SSH session."
echo ""
echo "Helpful commands:"
echo "  Live log tail:   tail -f $LOG_FILE"
echo "  Check GPU:       watch -n 1 nvidia-smi"
echo "  Check process:   ps -p $PID"
echo "  Stop training:   kill $PID"
echo "  ClearML web:     https://app.clear.ml"
echo ""
echo "Saved Charts & Diagrams location:"
echo "  All plots:       output/runs/latest/reports/*.png"
echo "  HTML reports:    output/runs/latest/reports/report.html"
echo "  Zipped bundle:   output/runs/latest/reports.zip"
echo "  Executed cells:  squirrel_detection_rfdetr_medium.ipynb"
echo "=========================================================="
