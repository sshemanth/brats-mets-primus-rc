#!/bin/bash
while true; do
  nn=$(ps -C nnUNetv2_train -o rss= | awk '{s+=$1; if($1>m)m=$1} END{printf "nnUNet total %d MB, largest %d MB, procs %d", s/1024, m/1024, NR}')
  echo "$(date '+%F %T') | $(free -m | awk '/Mem/{printf "used %d MB, avail %d MB, shared %d MB", $3, $7, $5}') | swap $(free -m | awk '/Swap/{print $3}') MB | $nn"
  sleep 300
done
