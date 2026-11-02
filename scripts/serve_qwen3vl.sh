#!/bin/bash
# Serve Qwen3-VL-8B-Instruct with vLLM 0.25.1 on one GPU (OpenAI-compatible API).
# Usage: scripts/serve_qwen3vl.sh <gpu-index> [bnb4|fp8]
#   bnb4: 4-bit bitsandbytes, port 8021, served as qwen3-vl-8b-bnb4 (needs ~12 GB free)
#   fp8 : FP8,                port 8022, served as qwen3-vl-8b-fp8  (needs ~15 GB free)
# --gpu-memory-utilization is set for a shared 32 GB card; raise it if the GPU is yours alone.
# --limit-mm-per-prompt 21 / --max-model-len 16384 covers Pilot 2 (1 query image + 20 lamps); Pilot 1 needs 10 / 12288.
set -e
GPU=${1:?gpu index}; MODE=${2:-bnb4}
cd "$(dirname "$0")/.."
if [ "$MODE" = fp8 ]; then Q=fp8; PORT=8022; UTIL=0.44; else Q=bitsandbytes; PORT=8021; UTIL=0.34; fi
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU vllm serve models/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b-$MODE --quantization $Q --host 127.0.0.1 --port $PORT \
  --gpu-memory-utilization $UTIL --max-model-len 16384 --max-num-seqs 1 --enforce-eager \
  --limit-mm-per-prompt '{"image": 21}' --mm-processor-kwargs '{"max_pixels": 409600}'
