#!/bin/bash
# Serve InternVL3-8B-Instruct (4-bit bitsandbytes) with vLLM 0.25.1 on one GPU (OpenAI-compatible API).
# Usage: scripts/serve_internvl3.sh <gpu-index>
#   port 8031, served as internvl3-8b-bnb4, needs ~10 GB free GPU memory.
# Each image is limited to ONE 448x448 tile (~256 tokens) so Pilot 2's 21 images fit in context.
# This is done with a config-only copy of the model folder (max_dynamic_patch 12 -> 1, weights symlinked);
# passing max_dynamic_patch through --mm-processor-kwargs makes vLLM 0.25.1 fail at startup.
set -e
GPU=${1:?gpu index}
cd "$(dirname "$0")/.."
SRC=models/InternVL3-8B-Instruct; DST=models/InternVL3-8B-Instruct-1tile
if [ ! -f $DST/config.json ]; then
  mkdir -p $DST
  for f in $SRC/*; do b=$(basename "$f"); [ "$b" = config.json ] || ln -sfn "$PWD/$f" "$DST/$b"; done
  python -c "import json; c=json.load(open('$SRC/config.json')); c['max_dynamic_patch']=1; json.dump(c, open('$DST/config.json','w'), indent=2)"
fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU vllm serve $DST \
  --served-model-name internvl3-8b-bnb4 --trust-remote-code --quantization bitsandbytes --host 127.0.0.1 --port 8031 \
  --gpu-memory-utilization 0.29 --max-model-len 16384 --max-num-seqs 1 --enforce-eager \
  --limit-mm-per-prompt '{"image": 21}'
