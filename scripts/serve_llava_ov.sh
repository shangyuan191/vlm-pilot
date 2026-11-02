#!/bin/bash
# Serve LLaVA-OneVision-Qwen2-7B (FP8) with vLLM 0.25.1 on one GPU (OpenAI-compatible API).
# Usage: scripts/serve_llava_ov.sh <gpu-index>
#   port 8032, served as llava-ov-7b-fp8, needs ~16 GB free GPU memory.
# FP8 because vLLM 0.25.1 has no bitsandbytes support for LlavaOnevisionForConditionalGeneration.
# The default anyres tiling costs ~2-4k tokens per image, so Pilot 2 (21 images) overflows the 32k context.
# A config-only copy of the model folder sets image_grid_pinpoints = [[384, 384]] (one 384x384 tile + base view,
# ~1.5k tokens per image; the longest Pilot 2 prompt measured 29,515 tokens). Weights are symlinked.
set -e
GPU=${1:?gpu index}
cd "$(dirname "$0")/.."
SRC=models/llava-onevision-qwen2-7b-ov-hf; DST=models/llava-onevision-qwen2-7b-ov-hf-384
if [ ! -f $DST/config.json ]; then
  mkdir -p $DST
  for f in $SRC/*; do b=$(basename "$f"); case "$b" in config.json|preprocessor_config.json) ;; *) ln -sfn "$PWD/$f" "$DST/$b";; esac; done
  python - "$SRC" "$DST" <<'PY'
import json, sys
src, dst = sys.argv[1:]
for name in ("config.json", "preprocessor_config.json"):
    c = json.load(open(f"{src}/{name}")); c["image_grid_pinpoints"] = [[384, 384]]
    json.dump(c, open(f"{dst}/{name}", "w"), indent=2)
PY
fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU vllm serve $DST \
  --served-model-name llava-ov-7b-fp8 --quantization fp8 --host 127.0.0.1 --port 8032 \
  --gpu-memory-utilization 0.46 --max-model-len 32768 --max-num-seqs 1 --enforce-eager \
  --limit-mm-per-prompt '{"image": 21}'
