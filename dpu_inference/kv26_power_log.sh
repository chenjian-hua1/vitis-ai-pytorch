#!/bin/sh
# 用法: ./power_log.sh 輸出檔 取樣間隔(秒)
# chmod +x power_log.sh
# ./power_log.sh idle.csv 0.1 &      # 先量 idle 幾秒當基準
# sleep 5; kill $!

# 計算平均功耗
# 1. 跑完C++就結束監控
# ./power_log.sh run.csv 0.1 & LOG_PID=$!; <your_app>; kill $LOG_PID;
# 2. 設定程式執行時間
# ./power_log.sh run.csv 0.1 & LOG_PID=$!; timeout 60 ./inference_full_ip yolo_kv26_split.xmodel; kill $LOG_PID
# awk -F, 'NR>1{s+=$2; if($2>m)m=$2; n++} END{printf "avg=%.3f W  peak=%.3f W  samples=%d\n", s/n/1e6, m/1e6, n}' run.csv
# 3. Benchmark
# ./kv26_power_log.sh run.csv 0.1 & LOG_PID=$!; BENCH_LB=ip timeout 60 ./benchmark_uyvyResizeQuantIP yolo_kv26_split.xmodel --iter 3000; kill $LOG_PID;

OUT=${1:-power.csv}
INTERVAL=${2:-0.1}

# 自動找 INA260 的 hwmon
for d in /sys/class/hwmon/hwmon*; do
  [ "$(cat $d/name)" = "ina260_u14" ] && HW=$d
done
[ -z "$HW" ] && { echo "找不到 ina260"; exit 1; }

echo "uptime_s,power_uW,current_mA,voltage_mV" > "$OUT"
while true; do
  t=$(cut -d' ' -f1 /proc/uptime)
  echo "$t,$(cat $HW/power1_input),$(cat $HW/curr1_input),$(cat $HW/in1_input)" >> "$OUT"
  sleep "$INTERVAL"
done