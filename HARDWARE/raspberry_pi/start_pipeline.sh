#!/bin/bash
echo "=========================================================="
echo " Starting Military ANC Hybrid Pipeline (Raspberry Pi AI) "
echo "=========================================================="

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Ensure Python dependencies are available
pip install -r requirements.txt --quiet

# Launch FPGA Bridge Service in Background
python3 fpga_bridge.py
