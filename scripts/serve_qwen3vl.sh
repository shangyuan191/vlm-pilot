#!/bin/bash
# Serve Qwen3-VL-8B-Instruct (4-bit bitsandbytes) with vLLM 0.25.1 on one GPU (OpenAI-compatible API).
# Usage: scripts/serve_qwen3vl.sh <gpu-index>
#   port 8021, served as qwen3-vl-8b-bnb4, needs ~12 GB free GPU memory.
# --gpu-memory-utilization is set for a shared 32 GB card; raise it if the GPU is yours alone.
# --limit-mm-per-prompt 21 / --max-model-len 16384 covers Pilot 2 (1 query image + 20 lamps); Pilot 1 needs 10 / 12288.
set -e
GPU=${1:?gpu index}
cd "$(dirname "$0")/.."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU vllm serve models/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b-bnb4 --quantization bitsandbytes --host 127.0.0.1 --port 8021 \
  --gpu-memory-utilization 0.34 --max-model-len 16384 --max-num-seqs 1 --enforce-eager \
  --limit-mm-per-prompt '{"image": 21}' --mm-processor-kwargs '{"max_pixels": 409600}'
