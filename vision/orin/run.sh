#!/usr/bin/env bash
# Keeps the Orin server up: restarts it within seconds if it ever dies, and logs why.
#   ./run.sh            full mode
#   LOW_MEM=1 ./run.sh  lightest mode (no big model; teaching loads on first use)
cd "$(dirname "$0")"
while true; do
  echo "=== start $(date +%T) ===" | tee -a server.log
  python3 -u navbelt_server.py 2>&1 | tee -a server.log
  code=${PIPESTATUS[0]}
  echo "=== server exited (code $code) at $(date +%T); restarting in 2 s ===" | tee -a server.log
  dmesg 2>/dev/null | grep -i -E "out of memory|killed process|oom" | tail -3 | tee -a server.log
  sleep 2
done
