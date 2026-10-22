# Dory1.0-4B serving image on the public vLLM nightly (fe4df3df) + the Dory model code of vLLM PR #60149 as a
# pure-Python overlay, plus per-request loop counts ("loops"), per-request n-gram draft caps ("spec_k") and the
# GB10 kernel overlay (mixed loop counts in one pass, per-request per-loop KV sizing, optional FP4 MLP prefill).
#   docker build -t dory-vllm:loops-speck-kern .
# Dockerfile.slim layers the same overlay on the Nemotron Deciffusion slim serving image instead (the published
# ghcr.io image); the Dory path is identical in both.
ARG BASE=vllm/vllm-openai:nightly-fe4df3df0eb703862c686b256e9fe44fa94c8945
FROM ${BASE}
USER root
COPY overlay /opt/dory-overlay
RUN bash /opt/dory-overlay/install.sh \
 && python3 -c "from vllm.model_executor.models.registry import ModelRegistry; assert 'DoryForCausalLM' in ModelRegistry.get_supported_archs()"
ENTRYPOINT []
