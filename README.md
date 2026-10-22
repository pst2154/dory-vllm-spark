# dory-vllm-spark

Fast serving for [NVIDIA Nemotron-Labs Dory1.0-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4)
(looped 4B, NVFP4) with vLLM on DGX Spark (GB10).

It adds to the Dory model code of vLLM PR [#60149](https://github.com/vllm-project/vllm/pull/60149):

- **Per-request loop count**: `"loops": 1-4` in the request body. Requests with different loop counts share a batch,
  and no restart is needed. Output matches a server started with that fixed count.
- **Per-request speculation cap**: `"spec_k": 0-8` caps n-gram (`ngram_gpu`) drafts for that request. A batch-size
  schedule speculates only at low concurrency.
- **GB10 kernel overlay**:
  - Mixed loop counts run in one pass, 1.9x faster on a 1/2/4 mix.
  - The per-loop KV cache is sized to each request's own loop count, so 1-2 loop traffic doesn't pay for the 4-loop
    reservation.
  - A faster single-request output layer.
  - Optional FP4 tensor-core MLP prefill (`FP4_PREFILL=1`): about 1.5x faster prefill, but it quantizes MLP
    activations to FP4 and lowers accuracy (ARC-Challenge 70.4 → 65.2), so it is off by default.

Upstreaming is tracked in vLLM draft PRs linked from #60149.

## Quick start (DGX Spark)

```bash
huggingface-cli download nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4 --local-dir ~/models/Dory1.0-4B-NVFP4
./run.sh 8000
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "dory-4b", "messages": [{"role": "user", "content": "Summarize: ..."}],
  "loops": 1, "spec_k": 8, "max_tokens": 128}'
```

`run.sh` pulls `ghcr.io/pst2154/dory-vllm:slim-rc7-loops-speck-kern-arm64`. Alternatively, build the image yourself on
the public vLLM nightly:

```bash
docker build -t dory-vllm:loops-speck-kern .
IMAGE=dory-vllm:loops-speck-kern ./run.sh 8000
```

Its defaults:

| Setting | Value |
|---|---|
| Loops | 2 by default, up to 4 per request |
| KV cache | FP8, prefix caching on |
| Attention backend | FlashInfer |
| Concurrent sequences | 128 |
| n-gram speculation | 8 drafts at 1-2 running requests, 5 at 3-4, off from 5 |
| FP4 MLP prefill | Off (`FP4_PREFILL=1` to turn it on; lowers accuracy) |

## Options

Server settings: environment variables for `run.sh`.

| Variable | Default | Meaning |
|---|---|---|
| `IMAGE` | `ghcr.io/pst2154/dory-vllm:slim-rc7-loops-speck-kern-arm64` | Serving image |
| `MODEL_DIR` | `~/models/Dory1.0-4B-NVFP4` | Local model directory |
| `LOOPS` | `2` | Loop count for requests that don't send `"loops"` |
| `MAX_LOOPS` | `4` | Highest loop count a request may ask for |
| `SPEC_KMAX` | `8` | Maximum n-gram drafts per step; `0` turns speculation off |
| `SPEC_SCHEDULE` | `[[1,2,8],[3,4,5],[5,1000000,0]]` | Drafts by number of running requests, as `[min, max, drafts]` |
| `FP4_PREFILL` | `0` | `1` runs MLP prefill on FP4 tensor cores: about 1.5x faster prefill, lower accuracy (ARC-Challenge 70.4 → 65.2) |
| `ATTN` | `FLASHINFER` | Attention backend (`TRITON_ATTN` also works) |
| `MAX_LEN` | `32768` | Maximum context length |
| `GPU_UTIL` | `0.40` | vLLM `--gpu-memory-utilization`, a share of unified memory |
| `MAX_SEQS` | `128` | Maximum concurrent sequences |

Per-request fields, sent in the chat-completions body:

| Field | Values | Meaning |
|---|---|---|
| `loops` | 1–4 | Loop count for this request, with no restart |
| `spec_k` | 0–8 | Cap on this request's n-gram drafts; the schedule above still applies |

## Benchmarks (llama-benchy 0.4.0, one GB10)

Spark Arena v2 profile (pp2048, tg128, concurrency 1/2/5/10, prefix caching, 3 runs), tok/s:

| Configuration | tg128 c1 | tg128 c10 (total / peak) | pp2048 c1 |
|---|---|---|---|
| 1 loop, no speculation | 76.3 | 275 / 580 | 7,307 |
| 1 loop, n-gram speculation | 122 (±68, prompt-dependent) | 264 / 665 | 7,297 |
| 2 loops, n-gram speculation | 114 (±48) | 179 / 488 | 4,682 |
| **1 loop, `run.sh` defaults** (FlashInfer, speculation at 1-4 running requests) | **116 (±60)** | **274 / 580** | 7,442 |

The first three rows used the Triton attention backend with speculation up to 12 concurrent requests. Speculating only
at low concurrency keeps the single-stream gain and matches no-speculation throughput under load. The `run.sh` row
is the current default (FP4 MLP prefill off).

Full tables are in [`benchmarks/`](benchmarks):

- `arena-*.md`: Spark Arena v2 profile;
- `benchy-*.md`: pp512/2048, tg32/128, depth 0/4k/8k, concurrency 1/8/32.

## Layout

| Path | Contents |
|---|---|
| `overlay/` | The Dory model and config from PR #60149, the per-request loops and spec_k code, and `install.sh`, which applies them to an installed vLLM |
| `Dockerfile` | The public vLLM nightly `fe4df3df` plus the overlay |
| `Dockerfile.slim` | The same overlay on the Nemotron Deciffusion slim serving image (the published image; one image serves both models) |
| `run.sh` | Serves Dory on a Spark with the defaults above |

## License

Apache-2.0 (see LICENSE). The overlay derives from vLLM (Apache-2.0) and the Dory model code of PR #60149. The
Dory1.0-4B weights are not included; they are distributed by NVIDIA under the OpenMDW-1.1 license.
