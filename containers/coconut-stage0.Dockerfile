FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

ARG SOURCE_REVISION=UNPINNED
ARG COCONUT_REVISION=27273cb8cca4bb763c041a63b036d0c3b7cbbb48
ARG GPT2_REVISION=607a30d783dfa663caf39e06633721c8d4cfcd7e
ARG GSM_SOURCE_REVISION=e06a32ee5e4cd117171daeb4755d2a97ece62761
ARG GSM_TRAIN_SHA256=0a3909a9e7d8d2f7ad6b8c7b5608aa744988d835f9bf874d4cf06ca77df6bf8c
ARG GSM_TRAIN_SIZE=87805358

ENV LATENT_REASONING_SOURCE_REVISION=${SOURCE_REVISION}
ENV LATENT_REASONING_CONTAINER=coconut-stage0-v1
ENV COCONUT_REVISION=${COCONUT_REVISION}
ENV GPT2_REVISION=${GPT2_REVISION}
ENV GSM_SOURCE_REVISION=${GSM_SOURCE_REVISION}
ENV GSM_TRAIN_SHA256=${GSM_TRAIN_SHA256}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-coconut-stage0.txt ./
RUN python -m pip install --no-cache-dir -r requirements-coconut-stage0.txt

RUN git clone --filter=blob:none https://github.com/facebookresearch/coconut.git /opt/coconut \
    && git -C /opt/coconut checkout "${COCONUT_REVISION}" \
    && test "$(git -C /opt/coconut rev-parse HEAD)" = "${COCONUT_REVISION}" \
    && printf '%s\n' "${COCONUT_REVISION}" > /opt/coconut/SOURCE_REVISION \
    && cp /opt/coconut/LICENSE /opt/coconut/LICENSE.upstream \
    && rm -rf /opt/coconut/.git

COPY tools/hydrate_pinned_lfs.py /usr/local/bin/hydrate_pinned_lfs.py

RUN mkdir -p /opt/data-receipts /tmp/icot \
    && rmdir /tmp/icot \
    && python /usr/local/bin/hydrate_pinned_lfs.py \
         --repository da03/Internalize_CoT_Step_by_Step \
         --revision "${GSM_SOURCE_REVISION}" \
         --lfs-path data/gsm8k/train.txt \
         --oid "${GSM_TRAIN_SHA256}" \
         --size "${GSM_TRAIN_SIZE}" \
         --include LICENSE \
         --include data/gsm8k/valid.txt \
         --include data/gsm8k/test.txt \
         --checkout-dir /tmp/icot \
         --receipt /opt/data-receipts/gsm-lfs-hydration.json \
    && mkdir -p /opt/licenses \
    && cp /tmp/icot/LICENSE /opt/licenses/Internalize_CoT_Step_by_Step.LICENSE \
    && cp /tmp/icot/data/gsm8k/train.txt /opt/coconut/data/gsm_train.txt \
    && cp /tmp/icot/data/gsm8k/valid.txt /opt/coconut/data/gsm_valid.txt \
    && cp /tmp/icot/data/gsm8k/test.txt /opt/coconut/data/gsm_test.txt \
    && cd /opt/coconut \
    && python preprocessing/gsm_icot.py train \
    && python preprocessing/gsm_icot.py valid \
    && python preprocessing/gsm_icot.py test \
    && python - <<'PY'
import hashlib
import json
from pathlib import Path

root=Path("/opt/coconut/data")
expected={
    "train":"93a266b1f7425a00463bb775de2946f9b30f76514944b5456bbffceb50b5de30",
    "valid":"1833a3479186e4e6c0183aad32fb7ee343479c958f2a1b57631ec22f5ee3aa7f",
    "test":"46ce5b0c44fde0bfc8cf1e7492aa640b96111d9dcb80a54340d9911141a9caa9",
}
receipt={"schema_version":1,"processed":{}}
for split,digest in expected.items():
    path=root/f"gsm_{split}.json"
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"{split} processed digest mismatch: {actual} != {digest}")
    receipt["processed"][split]={
        "sha256":actual,
        "size_bytes":path.stat().st_size,
        "records":len(json.loads(path.read_text())),
    }
Path("/opt/data-receipts/gsm-processed.json").write_text(
    json.dumps(receipt,sort_keys=True)+"\n"
)
PY
RUN rm -rf /tmp/icot \
    /opt/coconut/data/gsm_train.txt \
    /opt/coconut/data/gsm_valid.txt \
    /opt/coconut/data/gsm_test.txt

RUN python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

revision=os.environ["GPT2_REVISION"]
target=Path("/opt/models/gpt2")
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
(target/"SOURCE_REVISION").write_text(revision+"\n")
PY

COPY configs/coconut-gsm-cot-stage0-2gpu.yaml /workspace/configs/coconut-gsm-cot-stage0-2gpu.yaml
COPY plans/coconut-gsm-cot-stage0-2gpu.json /workspace/plans/coconut-gsm-cot-stage0-2gpu.json
COPY tools/run_coconut_stage0.py /workspace/tools/run_coconut_stage0.py

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV WANDB_MODE=offline

ENTRYPOINT ["python", "/workspace/tools/run_coconut_stage0.py"]
