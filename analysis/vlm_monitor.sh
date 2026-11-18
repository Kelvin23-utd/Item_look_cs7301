#!/usr/bin/env bash
# Watch the lab workstation for the VLM naming experiment (analysis/VLM_NAMING_PLAN.md):
# who is on each GPU, whether one is free for us, and how our runs are going.
# Read-only: it starts nothing and stops nothing.
#
#   FB_WORKSTATION=<ssh alias> analysis/vlm_monitor.sh          # refresh every 60 s
#   FB_WORKSTATION=<ssh alias> analysis/vlm_monitor.sh 300      # every 5 minutes
#   FB_WORKSTATION=<ssh alias> analysis/vlm_monitor.sh 0        # once
#
# The workstation's address and account are private: they come from the environment and
# your SSH config, never from this file. FB_WS_DIR (the same variable analysis/vlm_ws.py uses, for
# example '/data/$USER/fb_vlm' in single quotes) overrides the project folder there; the default is
# /data/<your account>/fb_vlm.
set -u
WS="${FB_WORKSTATION:?set FB_WORKSTATION to the SSH alias of the lab workstation}"
EVERY="${1:-60}"

while true; do
  # Connecting can take 10-60 s; one shared connection is kept for 10 minutes so refreshes are quick.
  OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ControlMaster=auto -o ControlPersist=10m \
        -o ControlPath="$HOME/.ssh/fb-monitor-%C" "$WS" "FB_WS_DIR=\"${FB_WS_DIR:-}\" bash -s" <<'REMOTE' 2>&1
me=$(id -un)
dir=${FB_WS_DIR:-/data/$me/fb_vlm}
echo "== $(date '+%a %b %d %H:%M:%S') workstation time"
echo
echo "GPUs (free for us = no other account has a process on it):"
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
free=""
while IFS=, read -r idx uuid used total util; do
  uuid=$(echo "$uuid" | tr -d ' '); idx=$(echo "$idx" | tr -d ' ')
  mine=0; minemem=0; others=""
  while IFS=, read -r u pid mem; do
    [ "$(echo "$u" | tr -d ' ')" = "$uuid" ] || continue
    pid=$(echo "$pid" | tr -d ' ')
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ "$owner" = "$me" ]; then mine=1; minemem=$((minemem + $(echo "$mem" | tr -dc '0-9')))
    else others="$others ${owner:-unknown}(pid $pid,$mem MiB, up$(ps -o etime= -p "$pid" 2>/dev/null))"; fi
  done <<< "$apps"
  # Shared = a small model of ours runs beside another account's idle process, under a memory cap.
  if [ -n "$others" ] && [ "$mine" = 1 ]; then state="SHARED, ours $minemem MiB beside other account:$others"; shared=1
  elif [ -n "$others" ]; then state="BUSY, other account:$others"
  elif [ "$mine" = 1 ]; then state="OURS, running ($minemem MiB)"
  else state="FREE"; free="$free $idx"; fi
  printf "  GPU %s: %5s /%6s MiB, %3s%% util  ->  %s\n" "$idx" "$(echo $used)" "$(echo $total)" "$(echo $util)" "$state"
done < <(nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
if [ -n "$free" ]; then echo "  => fully free now: GPU$free"
elif [ -n "${shared:-}" ]; then echo "  => none fully free; a small model of ours is running in shared mode"
else echo "  => none fully free (see the status below for what the queue is doing)"; fi
echo
echo "Our processes (model servers, runners, waiters):"
ps -u "$me" -o pid=,etime=,pcpu=,rss=,args= | grep -E "fb_vlm|vllm|vlm_" | grep -v -E "grep|bash -s" | cut -c1-150 | sed 's/^/  /' | grep . || echo "  none"
echo
if [ -d "$dir" ]; then
  if [ -f "$dir/runs/STATUS" ]; then echo "Status file ($(date -r "$dir/runs/STATUS" '+%H:%M:%S')):"; sed 's/^/  /' "$dir/runs/STATUS" | tail -10; echo; fi
  echo "Runs in $(basename "$dir")/runs (newest first):"
  ls -1t "$dir/runs" 2>/dev/null | grep -v '^STATUS$' | head -10 | while read -r r; do
    printf "  %-44s %s\n" "$r" "$(date -r "$dir/runs/$r" '+%b %d %H:%M')"
  done
  log=$(find "$dir/runs" -type f \( -name '*.log' -o -name '*.jsonl' \) -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  if [ -n "$log" ]; then
    echo
    echo "Newest log: ${log#$dir/} (changed $(date -r "$log" '+%H:%M:%S'))"
    tail -5 "$log" | cut -c1-160 | sed 's/^/  /'
  fi
  echo
  df -h "$dir" | awk 'NR==2{print "Disk: " $4 " free on " $6 " (" $5 " used)"}'
else
  echo "Project folder $dir not found."
fi
REMOTE
  )
  if [ "$EVERY" != 0 ] && [ -t 1 ]; then clear; fi
  echo "$OUT"
  [ "$EVERY" = 0 ] && break
  echo
  echo "(refreshing every ${EVERY}s; Ctrl-C to stop)"
  sleep "$EVERY"
done
