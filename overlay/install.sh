#!/bin/bash
# install vLLM PR #60149 (Dory) as an overlay into the vLLM inside this container (pure-Python files + 3 registry lines)
set -e
O=$(dirname $0); VP=${VP:-$(python3 -c "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('vllm').origin))")}   # VP=<site-packages>/vllm to patch an unpacked image from outside; no vllm import: works on CPU-only build nodes
cp $O/dory_model.py $VP/model_executor/models/dory.py
cp $O/dory_config.py $VP/transformers_utils/configs/dory.py
cp $O/dory_spec.py $VP/v1/core/sched/dory_spec.py
grep -q DoryForCausalLM $VP/model_executor/models/registry.py || sed -i '0,/"LlamaForCausalLM": ("llama", "LlamaForCausalLM"),/s//"LlamaForCausalLM": ("llama", "LlamaForCausalLM"),\n    "DoryForCausalLM": ("dory", "DoryForCausalLM"),/' $VP/model_executor/models/registry.py
grep -q 'dory="DoryConfig"' $VP/transformers_utils/config.py || sed -i '0,/^\( *\)deepseek_vl_v2="DeepseekVLV2Config",/s//&\n\1dory="DoryConfig",/' $VP/transformers_utils/config.py
grep -q DoryConfig $VP/transformers_utils/configs/__init__.py || python3 - $VP/transformers_utils/configs/__init__.py <<'PY'
import sys,re; p=sys.argv[1]; s=open(p).read()
s=s.replace('_CLASS_TO_MODULE: dict[str, str] = {', '_CLASS_TO_MODULE: dict[str, str] = {\n    "DoryConfig": "vllm.transformers_utils.configs.dory",',1)
s=s.replace('__all__ = [', '__all__ = [\n    "DoryConfig",',1)
open(p,'w').write(s)
PY
# Per-request loops: a request field `loops` (extra_body {"loops": N}, or vllm_xargs {"loops": N}) reaches
# SamplingParams.extra_args; the input processor validates it and salts the prefix cache per loop count.
python3 - $VP <<'PY'
import sys; vp=sys.argv[1]
def patch(path, anchor, new, marker):
    s=open(path).read()
    if marker in s: return
    assert s.count(anchor)==1, (path, anchor)
    open(path,'w').write(s.replace(anchor, new))
x='        extra_args: dict[str, Any] = self.vllm_xargs if self.vllm_xargs else {}\n'
y=x+'        if (self.model_extra or {}).get("loops") is not None:  # dory per-request loops\n            extra_args = {**extra_args, "loops": self.model_extra["loops"]}\n'
for f in ("entrypoints/openai/chat_completion/protocol.py", "entrypoints/openai/completion/protocol.py"):
    patch(f"{vp}/{f}", x, y, "dory per-request loops")
x='            cache_salt=decoder_input.get("cache_salt"),\n'
y='            cache_salt=cache_salt,\n'
f=f"{vp}/v1/engine/input_processor.py"
patch(f, x, y, "dory per-request loops")
x='        return EngineCoreRequest(\n            request_id=request_id,\n            prompt_token_ids=prompt_token_ids,\n'
y=('        cache_salt = decoder_input.get("cache_salt")\n'
   '        if getattr(self.model_config.hf_config, "model_type", None) == "dory":  # dory per-request loops\n'
   '            from vllm.transformers_utils.configs.dory import dory_request_loops\n'
   '            cache_salt = dory_request_loops(self.model_config.hf_config, sampling_params, cache_salt)\n'+x)
patch(f, x, y, "dory per-request loops")
x='            extra_args = {**extra_args, "loops": self.model_extra["loops"]}\n'
y=x+'        if (self.model_extra or {}).get("spec_k") is not None:  # dory per-request spec_k\n            extra_args = {**extra_args, "spec_k": self.model_extra["spec_k"]}\n'
for f in ("entrypoints/openai/chat_completion/protocol.py", "entrypoints/openai/completion/protocol.py"):
    patch(f"{vp}/{f}", x, y, "dory per-request spec_k")
# per-request spec_k in the async scheduler: each request's placeholder list has its own length (dory_spec.py)
f=f"{vp}/v1/core/sched/async_scheduler.py"
x='            request.spec_token_ids = self._spec_token_placeholders\n'
y=('            _k = dory_spec.request_k(request, len(self._spec_token_placeholders))  # dory per-request spec_k\n'
   '            request.spec_token_ids = (self._spec_token_placeholders if _k >= len(self._spec_token_placeholders)\n'
   '                                      else self._spec_token_placeholders[:_k])\n')
patch(f, x, y, "dory per-request spec_k")
x='from vllm.v1.core.sched.scheduler import Scheduler\n'
patch(f, x, x+'from vllm.v1.core.sched import dory_spec  # dory per-request spec_k\n', "import dory_spec")
# sync scheduling: truncate the drafts a request gets to its spec_k
f=f"{vp}/v1/core/sched/scheduler.py"
x='            request.spec_token_ids = self.structured_output_manager.validate_tokens(\n                request, spec_token_ids\n            )\n'
y=x+('            from vllm.v1.core.sched import dory_spec  # dory per-request spec_k\n'
     '            _k = dory_spec.request_k(request, len(request.spec_token_ids))\n'
     '            request.spec_token_ids = request.spec_token_ids[:_k]\n')
patch(f, x, y, "dory per-request spec_k")
# (new decode requests are padded to K only without a batch-size schedule; the server always runs one, so no change there)
# adaptive per-request K: the scheduler reports each request's verify outcome to dory_spec.observe
f=f"{vp}/v1/core/sched/scheduler.py"
x='                num_rejected = num_draft_tokens - num_accepted\n'
y=x+('                from vllm.v1.core.sched import dory_spec  # dory adaptive spec_k\n'
     '                dory_spec.observe(request, num_draft_tokens, num_accepted)\n')
patch(f, x, y, "dory adaptive spec_k")
PY
grep -c "dory per-request spec_k" $VP/entrypoints/openai/chat_completion/protocol.py $VP/v1/core/sched/async_scheduler.py $VP/v1/core/sched/scheduler.py
grep -c "dory per-request loops" $VP/entrypoints/openai/chat_completion/protocol.py $VP/entrypoints/openai/completion/protocol.py $VP/v1/engine/input_processor.py
grep -c Dory $VP/model_executor/models/registry.py $VP/transformers_utils/config.py $VP/transformers_utils/configs/__init__.py
