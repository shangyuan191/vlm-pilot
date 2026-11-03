#!/bin/bash
# Serve Qwen3-VL-32B-Instruct (4-bit bitsandbytes) with vLLM 0.25.1 on ONE GPU (OpenAI-compatible API).
# Usage: scripts/serve_qwen3vl_32b.sh <gpu-index>
#   port 8041, served as qwen3-vl-32b-bnb4, needs ~26 GB free GPU memory (a 32 GB card).
# Do not split it with --tensor-parallel-size 2: with bitsandbytes 4-bit, vLLM 0.25.1 then returns garbage
# (even for text-only prompts). Check with a plain question before running the pilots.
set -e
GPU=${1:?gpu index}
cd "$(dirname "$0")/.."
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU vllm serve models/Qwen3-VL-32B-Instruct \
  --served-model-name qwen3-vl-32b-bnb4 --quantization bitsandbytes --host 127.0.0.1 --port 8041 \
  --gpu-memory-utilization 0.78 --max-model-len 12288 --max-num-seqs 1 --enforce-eager \
  --limit-mm-per-prompt '{"image": 21}' --mm-processor-kwargs '{"max_pixels": 409600}'
