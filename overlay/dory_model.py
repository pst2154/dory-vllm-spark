# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dory: normalized latent recurrent transformer."""

import math
import os
import weakref
from collections.abc import Iterable
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.sequence import IntermediateTensors

from .utils import maybe_prefix

logger = init_logger(__name__)
EPS = 1e-6
_LMHEAD_PAD = os.environ.get("DORY_LMHEAD_PAD", "1") == "1"
_LAZY_LOOP_KV = os.environ.get("DORY_LAZY_LOOP_KV", "1") == "1"
# Mixed loop counts in one batch. 0 (default): one pass per distinct count, bit-exact
# vs each request alone. 1: one masked pass at the largest count (~1.9x on mixed
# batches), but not bit-exact: torch.compile fuses the mask into the norm reductions,
# so mixed-batch outputs can drift from the fixed-count graphs after some tokens.
_MIXED_SINGLE_PASS = os.environ.get("DORY_MIXED_SINGLE_PASS", "0") == "1"


def justnorm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """NGPT normalization: x * rsqrt(sum(x^2) + eps), fp32 accumulation."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).sum(dim=dim, keepdim=True) + EPS)
    return y.to(x.dtype)


def slerp_norm(a: torch.Tensor, b: torch.Tensor, lr: torch.Tensor) -> torch.Tensor:
    ya, yb = justnorm(a).float(), justnorm(b).float()
    return justnorm(ya + lr.float() * (yb - ya)).to(a.dtype)


def slerp_alpha(a, b, alpha, alpha_scale):
    return slerp_norm(a, b, (alpha.float() * alpha_scale).abs())


class DoryAttention(nn.Module):
    """GQA attention with nGPT qk-norm, output gate, and fngpt justnorm."""

    def __init__(
        self, config, layer_idx, cache_config, quant_config, prefix, num_loops=1
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        std = config.initializer_range

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.g_proj = ColumnParallelLinear(
            self.hidden_size,
            self.q_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.sqk = nn.Parameter(torch.empty(self.q_size))
        self.s_gate = nn.Parameter(torch.empty(self.q_size))
        self._sqk_factor = self.head_dim**0.5 / std
        g0 = config.attention_output_gate_logit_scale_init
        if g0 is None:
            g0 = config.hidden_size**0.5
        self._s_gate_factor = g0**0.5 / std

        layer_type = config.layer_types[layer_idx]
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters[layer_type],
            is_neox_style=True,
        )

        self.per_loop_kv = config.recurrent_kv_cache_mode == "per_loop"
        # Match LoopCoder's virtual layer indices for independent loop caches.
        # Projection prefixes keep the physical layer index to share weights.
        self.attn = nn.ModuleList()
        for loop_idx in range(num_loops if self.per_loop_kv else 1):
            unique_layer_idx = loop_idx * config.num_hidden_layers + layer_idx
            unique_prefix = prefix.replace(
                f".layers.{layer_idx}.", f".layers.{unique_layer_idx}."
            )
            self.attn.append(
                Attention(
                    self.num_heads,
                    self.head_dim,
                    scale=1.0 / math.sqrt(self.head_dim),
                    num_kv_heads=self.num_kv_heads,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    per_layer_sliding_window=(
                        config.sliding_window
                        if layer_type == "sliding_attention"
                        else None
                    ),
                    prefix=f"{unique_prefix}.attn",
                )
            )
            if loop_idx > 0 and _LAZY_LOOP_KV:
                _gate_kv_spec(self.attn[-1], loop_idx + 1)

    def forward(self, positions, hidden_states, loop_idx=0):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate, _ = self.g_proj(hidden_states)

        q, k = self.rotary_emb(positions, q, k)

        # nGPT qk-norm AFTER rope: q scaled by (sqk*f)^2 post-justnorm, k plain justnorm
        t = q.shape[0]
        q = q.view(t, self.num_heads, self.head_dim)
        sqk = (self.sqk.float() * self._sqk_factor).view(
            1, self.num_heads, self.head_dim
        )
        q = (sqk.square() * justnorm(q).float()).to(q.dtype).view(t, self.q_size)
        k = k.view(t, self.num_kv_heads, self.head_dim)
        k = justnorm(k).view(t, self.kv_size)

        # In last-loop mode every loop writes this step's KV to the same cache,
        # overwriting the previous loop's. Loop i thus attends to loop-i KV for
        # this step's tokens and last-loop KV for earlier tokens.
        out = self.attn[loop_idx if self.per_loop_kv else 0](q, k, v)

        # output gate (sigmoid, pre-o_proj), then fngpt justnorm over np*hn
        sg = (self.s_gate.float() * self._s_gate_factor).square()
        out = (out.float() * torch.sigmoid(gate.float() * sg)).to(out.dtype)
        out = justnorm(out)
        out, _ = self.o_proj(out)
        return out


class DoryMLP(nn.Module):
    """SwiGLU with nGPT suv scaling and fngpt justnorm."""

    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.ffn = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.ffn] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.ffn,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.suv_gate = nn.Parameter(torch.empty(self.ffn))
        self.suv_up = nn.Parameter(torch.empty(self.ffn))
        self._suv_factor = config.hidden_size**0.5

    def forward(self, hidden_states):
        m, _ = self.gate_up_proj(hidden_states)
        g, u = m.chunk(2, dim=-1)
        g = (g.float() * (self.suv_gate.float() * self._suv_factor)).to(g.dtype)
        u = (u.float() * (self.suv_up.float() * self._suv_factor)).to(u.dtype)
        m = (F.silu(g.float()) * u.float()).to(m.dtype)
        m = justnorm(m)
        out, _ = self.down_proj(m)
        return out


class DoryDecoderLayer(nn.Module):
    """One physical layer: a single sublayer (attention OR mlp) + slerp residual."""

    def __init__(
        self, config, layer_idx, cache_config, quant_config, prefix, num_loops=1
    ):
        super().__init__()
        self.is_attention = config.layer_types[layer_idx] != "mlp"
        self._alpha_scale = config.ngpt_alpha_init / config.initializer_range
        if self.is_attention:
            self.mixer = DoryAttention(
                config,
                layer_idx,
                cache_config,
                quant_config,
                f"{prefix}.mixer",
                num_loops=num_loops,
            )
            self.attn_alpha = nn.Parameter(torch.empty(config.hidden_size))
        else:
            self.mixer = DoryMLP(config, quant_config, f"{prefix}.mixer")
            self.mlp_alpha = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, positions, hidden_states, loop_idx=0):
        residual = hidden_states
        if self.is_attention:
            out = self.mixer(positions, hidden_states, loop_idx=loop_idx)
            return slerp_alpha(residual, out, self.attn_alpha, self._alpha_scale)
        out = self.mixer(hidden_states)
        return slerp_alpha(residual, out, self.mlp_alpha, self._alpha_scale)


class DoryBackbone(nn.Module):
    """Weights + layers. Not compiled itself: the compiled units are the
    DoryLoopVariant modules below, one per loop count."""

    def __init__(self, config, cache_config, quant_config, prefix):
        super().__init__()
        self.config = config
        # The per-loop KV caches are sized for the largest per-request count.
        self.num_loops = config.max_recurrent_loops or config.n_recurrent_loops
        n_in = config.n_input_layers
        n_rec = config.n_recurrent_layers
        self.embeddings = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DoryDecoderLayer(
                    config,
                    i,
                    cache_config,
                    quant_config,
                    f"{prefix}.layers.{i}",
                    num_loops=self.num_loops if n_in <= i < n_in + n_rec else 1,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def run(self, input_ids, positions, inputs_embeds, num_loops, loop_counts=None):
        h = inputs_embeds if inputs_embeds is not None else self.embeddings(input_ids)
        c = self.config
        n_in, n_rec = c.n_input_layers, c.n_recurrent_layers

        for layer in self.layers[:n_in]:
            h = layer(positions, h)
        input_h = h

        state = input_h
        for i in range(num_loops):
            combined = justnorm(state + input_h)
            h = combined
            for layer in self.layers[n_in : n_in + n_rec]:
                h = layer(positions, h, loop_idx=i)
            if loop_counts is None or i == 0:
                state = justnorm(h)
            else:
                # mixed batch: a token whose request asked for <= i loops keeps its state
                # (its loop-i compute is discarded; its loop-i KV slots are never read,
                # since its own count never attends loop i and blocks are salted per count)
                state = torch.where((loop_counts > i).unsqueeze(-1), justnorm(h), state)

        h = state
        for layer in self.layers[n_in + n_rec :]:
            h = layer(positions, h)
        return h


class _DoryLoopVariantBase(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", core=None,
                 num_loops: int = 1):
        super().__init__()
        self.core = core  # shared: parameters are de-duplicated by identity
        self.num_loops = num_loops


def _make_loop_variant(num_loops: int):
    """A torch.compile unit running `num_loops` loops: the same graph as a server
    started with n_recurrent_loops=num_loops. Each variant needs its own forward
    qualname: vLLM keys the AOT compile cache on the forward's qualname."""
    name = f"DoryLoopVariantL{num_loops}"

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.core.run(input_ids, positions, inputs_embeds, self.num_loops)

    forward.__qualname__ = f"{name}.forward"
    cls = type(name, (_DoryLoopVariantBase,), {"forward": forward, "__qualname__": name})
    return support_torch_compile(cls)


def _make_masked_loop_variant(num_loops: int):
    """Mixed batches: ONE pass of `num_loops` (= the largest count in the batch) loops;
    loop i only updates the tokens whose request asked for more than i loops. Cost =
    the largest count, not the sum of the counts (the old grouped passes ran the input
    and output layers once per count and every count's loops)."""
    name = f"DoryMaskedLoopVariantL{num_loops}"

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        loop_counts: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.core.run(input_ids, positions, inputs_embeds, self.num_loops, loop_counts)

    forward.__qualname__ = f"{name}.forward"
    cls = type(name, (_DoryLoopVariantBase,), {"forward": forward, "__qualname__": name})
    return support_torch_compile(cls)


# --------------------------------------------------------------- per-request loops
# The model runner (V2) hands each request's SamplingParams.extra_args["loops"] to
# DoryModelState, which writes a per-token loop-count buffer and sets the step key.
# One count n in the batch: key (n,), the plain n-loop graph (the same graph as a
# server started with n loops). Several counts:
# - default (DORY_MIXED_SINGLE_PASS=0): key = the distinct counts, one pass per count.
#   During a pass only that count's tokens write KV (the others' slots are -1, like
#   padding) and each token keeps the hidden state of its own count's pass: bit-exact
#   vs running alone (deterministic mode); the step costs the sum of the counts.
# - DORY_MIXED_SINGLE_PASS=1: key (0, m), largest m, ONE masked m-loop pass: loop i
#   only updates tokens whose count is > i. The step costs m loops, but it is not
#   bit-exact (the mask changes the compiled norm kernels). A request's per-loop caches
# above its count are never read by it (and only get scratch writes from masked
# loops); its blocks are only reused by requests with the same count (prefix-cache
# salt per count).
# FULL CUDA graphs are captured once per step key (CudaGraphManager patch below).
_ACTIVE: "weakref.ref[DoryForCausalLM] | None" = None


def _active_model():
    return _ACTIVE() if _ACTIVE is not None else None


def _patch_cudagraph_manager():
    try:
        from vllm.config.compilation import CUDAGraphMode
        from vllm.v1.worker.gpu import cudagraph_utils as cgu
    except Exception:  # noqa: BLE001  (V1 runner only: default loops only)
        return
    mgr = cgu.CudaGraphManager
    if getattr(mgr, "_dory_patched", False):
        return
    orig_capture, orig_run, orig_release = mgr.capture, mgr.run_fullgraph, mgr.release_graphs

    def capture(self, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"):
        m = _active_model()
        if m is None:
            return orig_capture(self, create_forward_fn, progress_bar_desc)
        by_key, descs = {}, self._capture_descs
        try:
            for key in m.step_keys:
                m.force_key = key
                self.graphs = {}
                if len(key) > 1:
                    # mixed keys: PIECEWISE replays the single-count variants' graphs
                    if CUDAGraphMode.FULL not in descs:
                        continue
                    self._capture_descs = {CUDAGraphMode.FULL: descs[CUDAGraphMode.FULL]}
                orig_capture(self, create_forward_fn, f"{progress_bar_desc} [loops={key}]")
                self._capture_descs = descs
                by_key[key] = self.graphs
        finally:
            m.force_key = None
            self._capture_descs = descs
        self._dory_graphs = by_key
        self.graphs = by_key[m.default_key]

    def run_fullgraph(self, desc):
        by_key = getattr(self, "_dory_graphs", None)
        m = _active_model()
        if by_key is not None and m is not None:
            self.graphs = by_key[m.step_key]
        return orig_run(self, desc)

    def release_graphs(self):
        self._dory_graphs = None
        return orig_release(self)

    mgr.capture, mgr.run_fullgraph, mgr.release_graphs = capture, run_fullgraph, release_graphs
    mgr._dory_patched = True


def _dory_model_state_cls():
    from vllm.utils.torch_utils import async_tensor_h2d
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    class DoryModelState(DefaultModelState):
        def __init__(self, vllm_config, model, encoder_cache, device):
            super().__init__(vllm_config, model, encoder_cache, device)
            cfg = vllm_config.model_config.hf_config
            self.default_loops = cfg.n_recurrent_loops
            # Per-request loops (indexed by the runner's req_index) and the persistent
            # per-token buffer the graphs read (FULL graphs replay it in place).
            self.req_loops = np.full(self.max_num_reqs, self.default_loops, dtype=np.int32)
            self.loop_counts = torch.full((self.max_num_tokens,), self.default_loops,
                                          dtype=torch.int32, device=device)
            self.debug = os.environ.get("DORY_LOOPS_DEBUG") == "1"
            self.step_counts: dict[tuple[int, ...], int] = {}

        def add_request(self, req_index, new_req_data):
            super().add_request(req_index, new_req_data)
            sp = new_req_data.sampling_params
            extra = (sp.extra_args or {}) if sp is not None else {}
            n = extra.get("loops")
            self.req_loops[req_index] = self.default_loops if n is None else int(n)

        def prepare_inputs(self, input_batch, req_states):
            out = super().prepare_inputs(input_batch, req_states)
            n = input_batch.num_tokens_after_padding
            loops = self.req_loops[input_batch.idx_mapping_np]
            key = tuple(int(x) for x in np.unique(loops))
            if len(key) > 1:
                if _MIXED_SINGLE_PASS:
                    key = (0, key[-1])
                tok = np.full(n, key[-1], dtype=np.int32)       # padding rows: any count
                num = input_batch.num_scheduled_tokens[: input_batch.num_reqs]
                tok[: int(num.sum())] = np.repeat(loops, num)
                async_tensor_h2d(tok, out=self.loop_counts[:n])
            self.model.step_key = key
            out["loop_counts"] = self.loop_counts[:n]
            if self.debug:
                self.step_counts[key] = self.step_counts.get(key, 0) + 1
                if sum(self.step_counts.values()) % 200 == 0:
                    logger.info("dory loop steps (loop counts in the batch): %s", self.step_counts)
            return out

        def prepare_dummy_inputs(self, num_reqs, num_tokens):
            out = super().prepare_dummy_inputs(num_reqs, num_tokens)
            key = self.model.force_key or self.model.default_key
            self.model.step_key = key
            self.loop_counts[:num_tokens].fill_(key[-1])
            out["loop_counts"] = self.loop_counts[:num_tokens]
            return out

    return DoryModelState


class DoryForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        # hf_overrides can change architecture fields after config initialization.
        config.validate()
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Dory currently supports tensor_parallel_size=1 only")
        self.backbone = DoryBackbone(
            config,
            vllm_config.cache_config,
            vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )
        # Compiled units: one N-loop graph per per-request loop count (1..max).
        from vllm.compilation.backends import set_model_tag

        default, max_loops = config.n_recurrent_loops, self.backbone.num_loops
        _DEFAULT_LOOPS["n"] = default
        counts = range(1, max_loops + 1)
        # step keys: (n,) = every token runs n loops (plain graph); (0, m) = mixed
        # counts, largest m (one masked m-loop graph)
        # counts, largest m (one masked m-loop graph); or, by default, the distinct
        # counts (one pass per count, replaying the single-count graphs)
        if _MIXED_SINGLE_PASS:
            mixed_keys = [(0, m) for m in counts if m > 1]
        else:
            mixed_keys = [k for r in range(2, max_loops + 1) for k in combinations(counts, r)]
        self.step_keys = [(n,) for n in counts] + mixed_keys
        self.default_key = (default,)
        self.step_key = self.default_key
        self.force_key = None
        self._all_compiled = False
        variants = {}
        for n in counts:
            with set_model_tag(f"dory_loops{n}"):
                variants[str(n)] = _make_loop_variant(n)(
                    vllm_config=vllm_config,
                    prefix=maybe_prefix(prefix, f"loops{n}"),
                    core=self.backbone,
                    num_loops=n,
                )
        self.loop_variants = nn.ModuleDict(variants)
        masked = {}
        for n in counts if _MIXED_SINGLE_PASS else ():
            if n == 1:
                continue
            with set_model_tag(f"dory_masked{n}"):
                masked[str(n)] = _make_masked_loop_variant(n)(
                    vllm_config=vllm_config,
                    prefix=maybe_prefix(prefix, f"masked{n}"),
                    core=self.backbone,
                    num_loops=n,
                )
        self.masked_variants = nn.ModuleDict(masked)
        global _ACTIVE
        _ACTIVE = weakref.ref(self)
        _patch_cudagraph_manager()
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.lm_head.logit_scale = nn.Parameter(torch.empty(config.vocab_size))
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.backbone.embeddings(input_ids)

    def get_model_state_cls(self):
        return _dory_model_state_cls()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        loop_counts: torch.Tensor | None = None,
    ):
        key = self.step_key if loop_counts is not None else self.default_key
        if not self._all_compiled:
            # First forward = the engine's profiling run: compile every variant here, on
            # the same inputs. A variant compiled later (first called during CUDA graph
            # capture, smaller batch) gets other inductor size hints, hence other kernel
            # configs and other rounding than a server started with that loop count.
            self._all_compiled = True
            for n in self.loop_variants:
                if key != (int(n),):
                    self.loop_variants[n](input_ids, positions, inputs_embeds)
            lc = loop_counts
            if lc is None:
                lc = torch.ones(positions.shape[-1], dtype=torch.int32, device=positions.device)
            for n in self.masked_variants:
                if key != (0, int(n)):
                    self.masked_variants[n](input_ids, positions, lc, inputs_embeds)
        if len(key) == 1:
            return self.loop_variants[str(key[0])](input_ids, positions, inputs_embeds)
        if key[0] != 0:
            return self._grouped_passes(key, input_ids, positions, inputs_embeds, loop_counts)
        return self.masked_variants[str(key[-1])](input_ids, positions, loop_counts, inputs_embeds)

    def _grouped_passes(self, key, input_ids, positions, inputs_embeds, loop_counts):
        # Mixed batch, exact: one pass per loop count. During a pass only that count's
        # tokens write KV (the others' slots are -1, like padding), so every token's KV
        # and hidden state come from the graph of its own count, as when it runs alone.
        from vllm.forward_context import get_forward_context, is_forward_context_available

        maps = []
        if is_forward_context_available():
            sm = get_forward_context().slot_mapping
            if isinstance(sm, dict):
                maps = list({id(t): t for t in sm.values() if isinstance(t, torch.Tensor)}.values())
        saved = [t.clone() for t in maps]
        out = None
        for n in key:
            sel = loop_counts == n
            for t, o in zip(maps, saved):
                k = min(t.shape[0], sel.shape[0])
                t[:k].copy_(torch.where(sel[:k], o[:k], torch.full_like(o[:k], -1)))
            h = self.loop_variants[str(n)](input_ids, positions, inputs_embeds)
            out = h.clone() if out is None else torch.where(sel.unsqueeze(-1), h, out)
        for t, o in zip(maps, saved):
            t.copy_(o)
        return out

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        n = hidden_states.shape[0]
        if _LMHEAD_PAD and n < 4:
            # BF16 lm_head (131072 x 2560, 671 MB): at 1-3 rows cuBLAS picks a GEMV that
            # reads the weight at ~175 GB/s (3.9 ms on GB10); padded to 8 rows it runs the
            # tensor-core GEMM at ~240 GB/s (2.8 ms). Same math, other summation order.
            pad = hidden_states.new_zeros((8 - n, hidden_states.shape[1]))
            logits = self.logits_processor(self.lm_head, torch.cat([hidden_states, pad]))
            logits = logits[:n] if logits is not None else None
        else:
            logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None:
            logits = logits.float() * (
                self.lm_head.logit_scale.float() / self.config.initializer_range
            )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [
            # (target, artifact shard name, shard id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, w in weights:
            mapped = False
            for target, shard_name, shard_id in stacked:
                if f".{shard_name}." not in name:
                    continue
                pname = name.replace(f".{shard_name}.", f".{target}.")
                param = params[pname]
                if target == "qkv_proj":
                    heads = (
                        self.config.num_attention_heads
                        if shard_id == "q"
                        else self.config.num_key_value_heads
                    )
                    rows = heads * self.config.head_dim
                else:
                    rows = self.config.intermediate_size
                if isinstance(param, PerTensorScaleParameter):
                    shape = () if w.ndim == 0 else (1,)
                else:
                    # Preserve packed input columns and per-block scale columns.
                    shape = (rows, *param.shape[1:])
                self._check_weight_shape(name, w, shape)
                param.weight_loader(param, w, shard_id)
                loaded.add(pname)  # report the MODEL param name, not the artifact name
                mapped = True
                break
            if mapped:
                continue

            param = params[name]
            shape = (
                (self.config.vocab_size, self.config.hidden_size)
                if name in ("backbone.embeddings.weight", "lm_head.weight")
                else tuple(param.shape)
            )
            if isinstance(param, PerTensorScaleParameter) and w.ndim == 0:
                shape = ()
            self._check_weight_shape(name, w, shape)
            getattr(param, "weight_loader", default_weight_loader)(param, w)
            loaded.add(name)
        if os.environ.get("DORY_FP4_PREFILL") == "1":
            _install_fp4_prefill(self)
        return loaded

    @staticmethod
    def _check_weight_shape(name, weight, expected):
        # TP loaders may narrow oversized tensors; reject incompatible exports.
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"Dory weight {name}: expected shape {expected}, "
                f"got {tuple(weight.shape)}"
            )


# --------------------------------------------------------------- FP4 tensor-core prefill
# Opt-in (DORY_FP4_PREFILL=1). The checkpoint is W4A16 NVFP4, so vLLM runs Marlin (BF16
# activations x FP4 weights, dequantized to BF16 MMA). On GB10 Marlin is at the memory
# roofline for decode but is BF16-compute-bound for prefill. With DORY_FP4_PREFILL the
# linears run native FP4 x FP4 tensor-core GEMMs (CUTLASS, activations quantized to NVFP4)
# for batches of >= DORY_FP4_MIN_TOKENS tokens (prefill / long verify), and Marlin
# below (decode: unchanged numerics). Default: MLP linears only (DORY_FP4_MODULES=
# gate_up_proj,down_proj): prefill 1.5x faster, no-think verify agreement 141/200 vs
# 135/200 (all linears: 2x faster but 124-128/200). Costs a second copy of those FP4
# weights (+1.1 GB; all linears +1.9 GB).
# Activation global scale is static and needs no calibration: every Dory linear input is
# a justnorm'd (unit-norm) vector, so |x| <= 1 (layer 0: the embedding table's max |x|).
_FP4_LAYERS: dict = {}
_FP4_AMAX_HEADROOM = 1.125


def _dory_fp4_linear_impl(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    layer = _FP4_LAYERS[layer_name]
    if x.shape[0] >= layer._dory_fp4_min:
        from vllm import _custom_ops as ops

        xq, xs = ops.scaled_fp4_quant(x, layer._dory_a_gsi)
        return ops.cutlass_scaled_fp4_mm(
            xq, layer._dory_w4, xs, layer._dory_w4_sf, layer._dory_alpha, x.dtype
        )
    return layer._dory_orig_apply(layer, x, None)


def _dory_fp4_linear_fake(x: torch.Tensor, layer_name: str) -> torch.Tensor:
    n = _FP4_LAYERS[layer_name].output_size_per_partition
    return x.new_empty((x.shape[0], n))


_FP4_OP_REGISTERED = False


def _register_fp4_op():
    global _FP4_OP_REGISTERED
    if _FP4_OP_REGISTERED:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="dory_fp4_linear",
        op_func=_dory_fp4_linear_impl,
        mutates_args=[],
        fake_impl=_dory_fp4_linear_fake,
    )
    _FP4_OP_REGISTERED = True


def _dory_fp4_apply(self, layer, x, bias=None):
    """NVFP4 linear apply through the dory_fp4_linear custom op (Marlin below the
    token threshold, FP4 tensor cores at/above it)."""
    if not hasattr(layer, "_dory_w4"):
        return layer._dory_orig_apply(layer, x, bias)
    shape = x.shape
    out = torch.ops.vllm.dory_fp4_linear(x.reshape(-1, shape[-1]), layer._dory_name)
    if bias is not None:
        out = out + bias
    return out.view(*shape[:-1], out.shape[-1])


_HYBRID_CLS: dict = {}


def _install_fp4_prefill(model: nn.Module) -> None:
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        swizzle_blockscale,
    )

    min_tokens = int(os.environ.get("DORY_FP4_MIN_TOKENS", "128"))
    emb_amax = float(model.backbone.embeddings.weight.detach().abs().max().float())
    first = model.config.n_input_layers and "backbone.layers.0.mixer."
    _register_fp4_op()
    n_layers, nbytes = 0, 0
    only = tuple(x for x in os.environ.get("DORY_FP4_MODULES", "gate_up_proj,down_proj").split(",") if x)
    for name, mod in model.named_modules():
        if not isinstance(mod, LinearBase) or not hasattr(mod, "weight_scale_2"):
            continue
        if only and not name.endswith(only):
            continue
        if mod.weight.dtype != torch.uint8 or mod.weight_scale.dtype != torch.float8_e4m3fn:
            continue
        gs = mod.weight_scale_2.detach().float().reshape(-1)
        if torch.unique(gs).numel() != 1:
            logger.warning("dory fp4 prefill: %s has per-shard weight scales; skipped", name)
            continue
        bound = emb_amax if (first and name.startswith(first) and
                             not name.endswith(("o_proj", "down_proj"))) else 1.0
        bound *= _FP4_AMAX_HEADROOM
        mod._dory_w4 = mod.weight.detach().clone()
        mod._dory_w4_sf = swizzle_blockscale(mod.weight_scale.detach().clone())
        mod._dory_a_gsi = torch.tensor([448.0 * 6.0 / bound], dtype=torch.float32,
                                       device=gs.device)
        mod._dory_alpha = (gs[:1] * (bound / (448.0 * 6.0))).to(torch.float32)
        mod._dory_fp4_min = min_tokens
        qm = mod.quant_method
        base = type(qm)
        if base in _HYBRID_CLS.values():  # quant method object shared by layers
            base = base.__mro__[1]
        mod._dory_orig_apply = base.apply.__get__(qm)
        mod._dory_name = name
        # keep the quant method object (the loader still runs its Marlin repack); only
        # its class gains an apply override
        if base not in _HYBRID_CLS:
            _HYBRID_CLS[base] = type(f"DoryFp4{base.__name__}", (base,), {"apply": _dory_fp4_apply})
        qm.__class__ = _HYBRID_CLS[base]
        _FP4_LAYERS[name] = mod
        n_layers += 1
        nbytes += mod._dory_w4.numel() + mod._dory_w4_sf.numel()
    logger.info("dory fp4 prefill: %d linears use FP4 tensor cores for >= %d tokens "
                "(+%.2f GB weights, embedding amax %.3f)", n_layers, min_tokens,
                nbytes / 1e9, emb_amax)


# --------------------------------------------------------------- lazy per-loop KV
# Per-loop KV gives every recurrent attention layer one cache per loop, sized for
# max_recurrent_loops. Without this, every request reserves KV in all loops' caches
# (a 1- or 2-loop request pays for 4), which cuts the KV pool by up to 1.7x at the same
# memory. Here the cache of loop k (k >= 2) is its own KV cache group type ("gated":
# a FullAttention / SlidingWindow spec subclass carrying min_loops=k). Its manager
# gives a request whose loop count is < k only null blocks: no pool memory, no prefix
# cache entries; the worker's block table rows point at the null block, which only
# the discarded rows of a mixed-batch masked pass ever touch. Requests that run k
# loops allocate and cache exactly as before. The scheduler learns a request's loop
# count from SamplingParams.extra_args["loops"] (dory_request_loops validated it).
_REQ_LOOPS: dict = {}
_CUR = {"loops": None}
_DEFAULT_LOOPS = {"n": 2}


def _request_loops(request) -> int:
    sp = getattr(request, "sampling_params", None)
    extra = (getattr(sp, "extra_args", None) or {}) if sp is not None else {}
    n = extra.get("loops")
    return _DEFAULT_LOOPS["n"] if n is None else int(n)


def _gated_spec_classes():
    if "full" in _GATED:
        return _GATED["full"], _GATED["sw"]
    import copy
    from dataclasses import dataclass

    from vllm.v1.core import single_type_kv_cache_manager as stm
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
    from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    def _merge(cls, specs):
        if not all(spec == specs[0] for spec in specs[1:]):
            raise ValueError("dory gated KV specs differ (loop index)")
        return copy.deepcopy(specs[0])

    @dataclass(frozen=True, kw_only=True)
    class DoryLoopFullAttentionSpec(FullAttentionSpec):
        min_loops: int = 0
        merge = classmethod(_merge)

    @dataclass(frozen=True, kw_only=True)
    class DoryLoopSlidingWindowSpec(SlidingWindowSpec):
        min_loops: int = 0
        merge = classmethod(_merge)

    # base FullAttentionSpec.merge accepts subclasses: keep gated layers out of the
    # plain buckets (and different loops apart)
    orig_full_merge = FullAttentionSpec.merge.__func__
    if not getattr(FullAttentionSpec, "_dory_merge_patched", False):
        def full_merge(cls, specs):
            if len({getattr(sp, "min_loops", 0) for sp in specs}) > 1:
                raise ValueError("dory: gated and plain KV specs do not merge")
            return orig_full_merge(cls, specs)
        FullAttentionSpec.merge = classmethod(full_merge)
        FullAttentionSpec._dory_merge_patched = True

    def gated(self, request_id):
        loops = _REQ_LOOPS.get(request_id)
        return loops is not None and loops < self.kv_cache_spec.min_loops

    def make_manager(base):
        class M(base):
            @classmethod
            def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                                       block_pool, kv_cache_spec, drop_eagle_block,
                                       alignment_tokens, dcp_world_size=1, pcp_world_size=1):
                loops = _CUR["loops"]
                if loops is not None and loops < kv_cache_spec.min_loops:
                    n = (max_length - max_length % alignment_tokens) // kv_cache_spec.block_size
                    null = block_pool.null_block
                    return tuple([null] * n for _ in kv_cache_group_ids), n * kv_cache_spec.block_size
                return super().find_longest_cache_hit(
                    block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
                    drop_eagle_block, alignment_tokens, dcp_world_size, pcp_world_size)

            def get_num_blocks_to_allocate(self, request_id, *args, **kwargs):
                if gated(self, request_id):
                    return 0
                return super().get_num_blocks_to_allocate(request_id, *args, **kwargs)

            def add_local_computed_blocks(self, request_id, new_computed_blocks, *args, **kwargs):
                if gated(self, request_id):
                    blocks = self.req_to_blocks[request_id]
                    blocks.extend([self._null_block] * len(new_computed_blocks))
                    self.num_cached_block[request_id] = len(blocks)
                    return None
                return super().add_local_computed_blocks(request_id, new_computed_blocks, *args, **kwargs)

            def allocate_external_computed_blocks(self, request_id, *args, **kwargs):
                if gated(self, request_id):
                    return None
                return super().allocate_external_computed_blocks(request_id, *args, **kwargs)

            def allocate_new_blocks(self, request_id, num_tokens, *args, **kwargs):
                if gated(self, request_id):
                    blocks = self.req_to_blocks[request_id]
                    n = -(-num_tokens // self.block_size) - len(blocks)
                    if n <= 0:
                        return []
                    new = [self._null_block] * n
                    blocks.extend(new)
                    return new
                return super().allocate_new_blocks(request_id, num_tokens, *args, **kwargs)

            def cache_blocks(self, request, num_tokens, *args, **kwargs):
                if gated(self, request.request_id):
                    self.num_cached_block[request.request_id] = num_tokens // self.block_size
                    return None
                return super().cache_blocks(request, num_tokens, *args, **kwargs)

            def remove_skipped_blocks(self, request_id, *args, **kwargs):
                if gated(self, request_id):
                    return None
                return super().remove_skipped_blocks(request_id, *args, **kwargs)

            def pop_blocks_for_free(self, request_id):
                was_gated = gated(self, request_id)
                blocks = super().pop_blocks_for_free(request_id)
                return [] if was_gated else blocks

            def get_num_common_prefix_blocks(self, running_request_id):
                if self.kv_cache_spec.min_loops > 0:
                    return 0  # no cascade attention over gated groups
                return super().get_num_common_prefix_blocks(running_request_id)

        M.__name__ = M.__qualname__ = f"DoryLoop{base.__name__}"
        return M

    KVCacheSpecRegistry._ensure_registered()
    KVCacheSpecRegistry.register(DoryLoopFullAttentionSpec, make_manager(stm.FullAttentionManager),
                                 uniform_type_base_spec=DoryLoopFullAttentionSpec)
    KVCacheSpecRegistry.register(DoryLoopSlidingWindowSpec, make_manager(stm.SlidingWindowManager),
                                 uniform_type_base_spec=DoryLoopSlidingWindowSpec)
    _GATED["full"], _GATED["sw"] = DoryLoopFullAttentionSpec, DoryLoopSlidingWindowSpec
    _patch_kv_cache_manager()
    return _GATED["full"], _GATED["sw"]


_GATED: dict = {}


def _patch_kv_cache_manager():
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if getattr(KVCacheManager, "_dory_patched", False):
        return
    orig_get, orig_alloc, orig_free = (KVCacheManager.get_computed_blocks,
                                       KVCacheManager.allocate_slots, KVCacheManager.free)

    def get_computed_blocks(self, request):
        loops = _request_loops(request)
        _REQ_LOOPS[request.request_id] = loops
        _CUR["loops"] = loops
        try:
            return orig_get(self, request)
        finally:
            _CUR["loops"] = None

    def allocate_slots(self, request, *args, **kwargs):
        if request.request_id not in _REQ_LOOPS:
            _REQ_LOOPS[request.request_id] = _request_loops(request)
        return orig_alloc(self, request, *args, **kwargs)

    def free(self, request):
        try:
            return orig_free(self, request)
        finally:
            _REQ_LOOPS.pop(request.request_id, None)

    KVCacheManager.get_computed_blocks = get_computed_blocks
    KVCacheManager.allocate_slots = allocate_slots
    KVCacheManager.free = free
    KVCacheManager._dory_patched = True


def _gate_kv_spec(attn_layer, min_loops: int) -> None:
    """Make this per-loop attention layer report a gated KV spec (loop >= min_loops)."""
    import dataclasses

    orig = attn_layer.get_kv_cache_spec

    def get_kv_cache_spec(vllm_config):
        spec = orig(vllm_config)
        if spec is None:
            return spec
        full_cls, sw_cls = _gated_spec_classes()
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        if type(spec) is SlidingWindowSpec:
            cls = sw_cls
        elif type(spec) is FullAttentionSpec:
            cls = full_cls
        else:  # unknown spec type: keep eager reservation
            return spec
        fields = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec) if f.init}
        return cls(**fields, min_loops=min_loops)

    attn_layer.get_kv_cache_spec = get_kv_cache_spec
