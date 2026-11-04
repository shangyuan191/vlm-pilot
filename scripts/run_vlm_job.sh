#!/bin/bash
# Like run_vlm_when_free.sh, but the stages are chosen with env vars and a startup that loses its GPU memory
# to another process (another job grabbed the card between our check and vLLM's) is retried.
#
# Wait (up to 12 h per attempt) for a GPU with >= NEED_MB free on 7 checks in a row (~60 s, so we don't jump into the
# few seconds another queue leaves between two of its own models), serve one VLM there,
# run the chosen stages, shut the server down. Every run script resumes from its output file.
#
# Usage: [P1_KS="3 5"] [P2_RUNS="base 1 2"] scripts/run_vlm_job.sh <served-name> <model-dir> <port> <need-MB> [extra vllm args...]
#   P1_KS   : Pilot 1 K values ("" skips Pilot 1)
#   P2_RUNS : Pilot 2 runs; "base" = original candidate order, a number = run_pilot2.py --order-seed N ("" skips)
#   TP      : number of GPUs to split the model over (tensor parallel); each needs NEED_MB free (default 1)
#   GPUS    : space-separated GPU indices to consider (default: "0 1 3 4", the 5090s)
#   WAIT_H  : hours to wait for free GPUs per attempt (default 12)
#   P2_CAPS : caption sources for the fixed-caption check: run_pilot2.py --captions-from SRC, caption condition only ("" skips)
#   P2_CAP_RUNS : orders for P2_CAPS ("base" and/or seeds; default "base 1 2")
#   P2_PIPE : non-empty = also run pilot2/run_pipeline.py (per-image style scores + rule-based pick)
#   P2_JOBS : ";"-separated extra run_pilot2.py argument sets (e.g. "--conditions stylename;--lang en"), each run for P2_JOB_RUNS (default "base 1 2")
# Our own jobs take scripts/.vllm_start.lock from "GPU chosen" until vLLM is up, re-checking free memory after
# getting it, so two of our jobs never start on the same card at the same moment.
cd /mnt/raid0/home/skyler/vlm-diverse-retrieval
NAME=$1; MODEL=$2; PORT=$3; NEED=$4; shift 4
P1_KS=${P1_KS-3 5}; P2_RUNS=${P2_RUNS-base}; P2_CAPS=${P2_CAPS-}; P2_CAP_RUNS=${P2_CAP_RUNS-base 1 2}; TP=${TP:-1}; WAIT_H=${WAIT_H:-12}; GPUS=" ${GPUS:-0 1 3 4} "
ENV=$HOME/miniforge3/envs/adl_hw3_py311/bin
LOG=scripts/vllm_$NAME.log
PY=$PWD/.venv-author/bin/python
URL=http://127.0.0.1:$PORT/v1/chat/completions
echo "$(date -u +%FT%T) $NAME queued (P1_KS='$P1_KS' P2_RUNS='$P2_RUNS' P2_CAPS='$P2_CAPS')"

free_now() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' -v i=$1 '$1==i{print $3-$2}'; }
EXTRA=(); [ "$TP" -gt 1 ] && EXTRA=(--tensor-parallel-size $TP)
exec 9>scripts/.vllm_start.lock
for try in $(seq 1 20); do
  declare -A ok=(); gpu=""; deadline=$(( $(date +%s) + WAIT_H * 3600 ))
  while [ -z "$gpu" ] && [ $(date +%s) -lt $deadline ]; do
    ready=()
    while IFS=', ' read -r idx used total; do
      [[ "$GPUS" == *" $idx "* ]] || continue               # only the GPUs we were told to use
      free=$((total - used))
      if [ $free -ge $NEED ]; then ok[$idx]=$(( ${ok[$idx]:-0} + 1 )); else ok[$idx]=0; fi
      [ ${ok[$idx]} -ge 7 ] && ready+=($idx)
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
    if [ ${#ready[@]} -ge $TP ]; then
      flock 9                                                 # one of our jobs starting at a time
      gpu=""; n=0
      for i in "${ready[@]}"; do [ $n -lt $TP ] && [ $(free_now $i) -ge $NEED ] && { gpu=${gpu:+$gpu,}$i; n=$((n+1)); }; done
      if [ $n -lt $TP ]; then gpu=""; flock -u 9; declare -A ok=(); fi   # memory went away while we waited for the lock
    fi
    [ -z "$gpu" ] && sleep 10
  done
  [ -z "$gpu" ] && { echo "RESULT: $NAME no GPU freed up within $WAIT_H h"; exit 1; }
  echo "$(date -u +%FT%T) $NAME using GPU $gpu (try $try)"
  PATH=$ENV:$PATH CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu NCCL_P2P_DISABLE=1 $ENV/vllm serve $MODEL \
    --served-model-name $NAME --port $PORT --host 127.0.0.1 --max-num-seqs 1 --enforce-eager "${EXTRA[@]}" "$@" > $LOG 2>&1 &
  SERVER=$!
  until grep -q -E "Application startup complete|Engine core initialization failed|Error" $LOG; do sleep 5; kill -0 $SERVER 2>/dev/null || break; done
  flock -u 9
  grep -q "Application startup complete" $LOG && break
  kill $SERVER 2>/dev/null; sleep 10; pkill -P $SERVER 2>/dev/null
  if grep -q -E "Free memory on device|CUDA out of memory" $LOG; then
    echo "$(date -u +%FT%T) $NAME lost GPU memory during startup; retry in 2 min"; sleep 120; continue
  fi
  echo "RESULT: $NAME server failed to start on GPU $gpu"; grep -E "Error" $LOG | tail -3; exit 1
done
grep -q "Application startup complete" $LOG || { echo "RESULT: $NAME gave up after 20 startup attempts"; exit 1; }

# sanity check: a broken quantization/sharding can start fine and still emit garbage
ans=$(curl -s -m 300 $URL -H 'Content-Type: application/json' -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}],\"max_tokens\":10,\"temperature\":0}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>&1)
echo "$(date -u +%FT%T) $NAME sanity check answer: ${ans:0:60}"
if [[ "$ans" != *Paris* ]]; then
  echo "RESULT: $NAME failed the sanity check (no 'Paris' in the answer); not running the pilots"
  kill $SERVER; sleep 10; pkill -P $SERVER 2>/dev/null; exit 1
fi

for k in $P1_KS; do
  (cd pilot1 && $PY qwen_local_select.py --k $k --model $NAME --url $URL > ${NAME}_k$k.log 2>&1)
done
for r in $P2_RUNS; do
  if [ "$r" = base ]; then (cd pilot2 && $PY run_pilot2.py --model $NAME --url $URL > run_pilot2_$NAME.log 2>&1)
  else (cd pilot2 && $PY run_pilot2.py --model $NAME --url $URL --order-seed $r > run_pilot2_${NAME}_o$r.log 2>&1); fi
done
for src in $P2_CAPS; do for r in $P2_CAP_RUNS; do
  if [ "$r" = base ]; then (cd pilot2 && $PY run_pilot2.py --model $NAME --url $URL --captions-from $src > run_pilot2_${NAME}_cap-$src.log 2>&1)
  else (cd pilot2 && $PY run_pilot2.py --model $NAME --url $URL --captions-from $src --order-seed $r > run_pilot2_${NAME}_cap-${src}_o$r.log 2>&1); fi
done; done
IFS=";" read -ra JOBS <<< "${P2_JOBS-}"
for i in "${!JOBS[@]}"; do for r in ${P2_JOB_RUNS-base 1 2}; do o=""; [ "$r" = base ] || o="--order-seed $r"
  (cd pilot2 && $PY run_pilot2.py --model $NAME --url $URL ${JOBS[$i]} $o > run_pilot2_${NAME}_job${i}_$r.log 2>&1)
  echo "$(date -u +%FT%T) $NAME P2 job [${JOBS[$i]}] order $r: $(tail -n 1 pilot2/run_pilot2_${NAME}_job${i}_$r.log)"
done; done
if [ -n "${P2_PIPE-}" ]; then   # explicit per-image pipeline check (pilot2/run_pipeline.py)
  for t in 1 2 3; do (cd pilot2 && $PY run_pipeline.py --model $NAME --url $URL > run_pipeline_$NAME.log 2>&1); grep -q " 0 failed" pilot2/run_pipeline_$NAME.log && break; done
  echo "RESULT: $NAME pipeline: $(tail -n 1 pilot2/run_pipeline_$NAME.log)"
fi
kill $SERVER; sleep 10; pkill -P $SERVER 2>/dev/null

for k in $P1_KS; do f=pilot1/${NAME}_k${k}_selection.jsonl
  echo "RESULT: $NAME pilot1 K=$k ok $(grep -c '"status": "ok"' $f) failed $(grep -c failed_after $f)"; done
for r in $P2_RUNS; do t=$NAME; [ "$r" = base ] || t=${NAME}_o$r
  for c in full caption blind; do f=pilot2/${t}_$c.jsonl
    echo "RESULT: $t pilot2 $c ok $(grep -c '"status": "ok"' $f) failed $(grep -c failed_after $f)"; done; done
for src in $P2_CAPS; do for r in $P2_CAP_RUNS; do t=${NAME}_cap-$src; [ "$r" = base ] || t=${t}_o$r; f=pilot2/${t}_caption.jsonl
  echo "RESULT: $t pilot2 caption ok $(grep -c '"status": "ok"' $f 2>/dev/null) failed $(grep -c failed_after $f 2>/dev/null)"; done; done
