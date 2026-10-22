#!/bin/bash
# run.sh [port]: serve Dory1.0-4B NVFP4 on DGX Spark (GB10) with the measured-fastest defaults.
#   huggingface-cli download nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4 --local-dir ~/models/Dory1.0-4B-NVFP4
#   ./run.sh 8000
# Env: IMAGE, MODEL_DIR, LOOPS (default loops for requests without "loops": 2), MAX_LOOPS (4), SPEC_KMAX (8; 0 = off),
#      SPEC_SCHEDULE ([[1,2,8],[3,4,5],[5,1000000,0]]), FP4_PREFILL (0; 1 = on: ~1.5x prefill, lowers accuracy), FP8_PREFILL (0; 1 = FP8 W8A8 MLP prefill: ~1.33x prefill), MIXED_SINGLE_PASS (0; 1 = one masked pass for mixed loop counts: faster, not bit-exact), ATTN (FLASHINFER),
#      MAX_LEN (32768), GPU_UTIL (0.40), MAX_SEQS (128).
set -euo pipefail
PORT=${1:-8000}
IMAGE=${IMAGE:-ghcr.io/pst2154/dory-vllm:slim-rc7-loops-speck-kern-arm64}
MODEL_DIR=${MODEL_DIR:-$HOME/models/Dory1.0-4B-NVFP4}
LOOPS=${LOOPS:-2}; MAXL=${MAX_LOOPS:-4}; KMAX=${SPEC_KMAX:-8}; ATTN=${ATTN:-FLASHINFER}
# n-gram drafts by running batch size: speculate only at low concurrency (verification costs more than it saves when busy)
SCHED=${SPEC_SCHEDULE:-[[1,2,8],[3,4,5],[5,1000000,0]]}
SPEC=(); [ "$KMAX" -gt 0 ] && SPEC=(--speculative-config "{\"method\": \"ngram_gpu\", \"num_speculative_tokens\": $KMAX, \"prompt_lookup_max\": 4, \"prompt_lookup_min\": 2, \"num_speculative_tokens_per_batch_size\": $SCHED}")
exec docker run --rm --gpus all --ipc=host -p 127.0.0.1:$PORT:$PORT -v "$MODEL_DIR":/model:ro \
  -e DORY_LAZY_LOOP_KV=1 -e DORY_FP4_PREFILL=${FP4_PREFILL:-0} -e DORY_FP8_PREFILL=${FP8_PREFILL:-0} -e DORY_MIXED_SINGLE_PASS=${MIXED_SINGLE_PASS:-0} -e TRITON_F32_DEFAULT=ieee "$IMAGE" \
  vllm serve /model --served-model-name dory-4b --host 0.0.0.0 --port "$PORT" \
  --hf-overrides "{\"n_recurrent_loops\": $LOOPS, \"max_recurrent_loops\": $MAXL, \"recurrent_kv_cache_mode\": \"per_loop\"}" \
  --max-model-len "${MAX_LEN:-32768}" --max-num-seqs "${MAX_SEQS:-128}" --gpu-memory-utilization "${GPU_UTIL:-0.40}" \
  --enable-prefix-caching --async-scheduling --attention-backend "$ATTN" --kv-cache-dtype fp8 \
  --default-chat-template-kwargs '{"enable_thinking": false}' "${SPEC[@]}"
