FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

ARG SOURCE_REVISION=UNPINNED
ARG COCONUT_REVISION=27273cb8cca4bb763c041a63b036d0c3b7cbbb48
ARG GPT2_REVISION=607a30d783dfa663caf39e06633721c8d4cfcd7e

ENV LATENT_REASONING_SOURCE_REVISION=${SOURCE_REVISION}
ENV LATENT_REASONING_CONTAINER=coconut-gpu-smoke-v1
ENV COCONUT_REVISION=${COCONUT_REVISION}
ENV GPT2_REVISION=${GPT2_REVISION}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-coconut-gpu-smoke.txt ./
RUN python -m pip install --no-cache-dir -r requirements-coconut-gpu-smoke.txt

RUN git clone --filter=blob:none https://github.com/facebookresearch/coconut.git /opt/coconut \
    && git -C /opt/coconut checkout "${COCONUT_REVISION}" \
    && test "$(git -C /opt/coconut rev-parse HEAD)" = "${COCONUT_REVISION}" \
    && printf '%s\n' "${COCONUT_REVISION}" > /opt/coconut/SOURCE_REVISION \
    && rm -rf /opt/coconut/.git

RUN python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

revision = os.environ["GPT2_REVISION"]
target = Path("/opt/models/gpt2")
snapshot_download(
    repo_id="openai-community/gpt2",
    revision=revision,
    local_dir=target,
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
)
(target / "SOURCE_REVISION").write_text(revision + "\n")
PY

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY experiments ./experiments
COPY tools/run_replay.py ./tools/run_replay.py
COPY replays/coconut-reference-gpu-smoke.json ./replays/coconut-reference-gpu-smoke.json
COPY environments/pytorch251-cuda124-cudnn9-gpu.json ./environments/pytorch251-cuda124-cudnn9-gpu.json

ENTRYPOINT ["python", "tools/run_replay.py"]
