: ${MAX_BATCH_SIZE:=${1:-64}}
echo "MAX BATCH SIZE: $MAX_BATCH_SIZE"
BATCH_SIZES=()
POWER_OF_2=1
while [ $POWER_OF_2 -le $MAX_BATCH_SIZE ]; do
  BATCH_SIZES+=($POWER_OF_2)
  POWER_OF_2=$((POWER_OF_2 * 2))
done

echo "-b '$BS' -f '$BENCH_DIR/report-$BS.csv'"