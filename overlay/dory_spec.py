# SPDX-License-Identifier: Apache-2.0
"""Per-request speculative-token cap (installed as vllm/v1/core/sched/dory_spec.py by overlay/install.sh).

The server runs n-gram speculation (ngram_gpu) with a batch-size schedule (--speculative-config
num_speculative_tokens_per_batch_size: K per step by the number of running requests, e.g. 8 for 1-2, 5 for 3-4, 3 for
5-12, 0 above). A request may send "spec_k": n (0..8; extra_body {"spec_k": n} or vllm_xargs) and then verifies at most
min(n, step K) drafts per step; n = 0 turns speculation off for that request. Without the field a request uses
min(step K, DORY_SPEC_DEFAULT_CAP=5). Verification is the normal rejection sampler, so outputs keep the target model's
distribution (greedy: the target's argmax at every position).
The async scheduler gives each request a placeholder list of its own length; the V2 model runner already takes
per-request draft counts from scheduled_spec_decode_tokens, so drafts past a request's cap are never verified. A batch
whose requests use different counts is not a uniform decode batch and runs piecewise CUDA graphs for that step.

Adaptive draft length (DORY_SPEC_ADAPTIVE=1, default on): each request also keeps a running estimate of its per-position
acceptance a (accepted drafts / (accepted drafts + rejections), exponentially weighted over its recent verify steps; the
scheduler reports every step through observe()). Its next K is the number of draft positions whose survival
probability a^i stays above DORY_SPEC_MIN_SURVIVAL (0.3), so K grows while drafts keep being accepted and falls to 0
when they keep being rejected; at 0 it re-probes with one draft every DORY_SPEC_PROBE_EVERY (8) steps. The result is
min(adaptive K, spec_k cap, step K): the cap and the batch-size schedule always still apply."""
import math
import os

DEFAULT_CAP = int(os.environ.get("DORY_SPEC_DEFAULT_CAP", "5"))
MODE = os.environ.get("DORY_SPEC_ADAPTIVE", "0")   # 0 off, 1 per request, batch = one K for the whole step
ADAPTIVE = MODE in ("1", "batch")
_GLOBAL = [0.0, 0.0, 0]
MIN_SURVIVAL = float(os.environ.get("DORY_SPEC_MIN_SURVIVAL", "0.3"))
PROBE_EVERY = int(os.environ.get("DORY_SPEC_PROBE_EVERY", "8"))
DECAY = float(os.environ.get("DORY_SPEC_DECAY", "0.7"))      # weight of the history per observed verify step


def _cap(request) -> int:
    sp = getattr(request, "sampling_params", None)
    extra = (sp.extra_args or {}) if sp is not None else {}
    v = extra.get("spec_k")
    try:
        return DEFAULT_CAP if v is None else int(v)
    except (TypeError, ValueError):
        return DEFAULT_CAP


def _adaptive_k(request, cap: int) -> int:
    st = _GLOBAL if MODE == "batch" else getattr(request, "_dory_spec", None)
    if st is None:          # no observation yet: start at the cap
        return cap
    acc, rej, idle = st
    a = acc / (acc + rej) if acc + rej > 0 else 0.5
    if a <= 0.0:
        k = 0
    elif a >= 0.999:
        k = cap
    else:
        k = int(math.log(MIN_SURVIVAL) / math.log(a))
    if k <= 0:
        st[2] = idle + 1
        if st[2] >= PROBE_EVERY:  # re-probe so a request can recover
            st[2] = 0
            return 1
        return 0
    return min(k, cap)


def request_k(request, step_k: int) -> int:
    cap = max(0, min(_cap(request), step_k))
    if not ADAPTIVE or cap == 0:
        return cap
    return max(0, min(cap, _adaptive_k(request, cap)))


def observe(request, num_draft_tokens: int, num_accepted: int) -> None:
    """One verify step of `request`: num_accepted of its num_draft_tokens drafts were accepted (a prefix)."""
    if not ADAPTIVE or num_draft_tokens <= 0:
        return
    if MODE == "batch":
        st = _GLOBAL
    else:
        st = getattr(request, "_dory_spec", None)
        if st is None:
            st = [0.0, 0.0, 0]
            request._dory_spec = st
    st[0] = DECAY * st[0] + num_accepted
    st[1] = DECAY * st[1] + (1.0 if num_accepted < num_draft_tokens else 0.0)
