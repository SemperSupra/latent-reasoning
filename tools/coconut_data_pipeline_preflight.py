#!/usr/bin/env python3
"""Preflight exact Coconut GSM preprocessing/dataset/collator path on CPU."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer


COCONUT_REVISION = "27273cb8cca4bb763c041a63b036d0c3b7cbbb48"
DATA_REVISION = "e06a32ee5e4cd117171daeb4755d2a97ece62761"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset_module(root: Path):
    path=root/"dataset.py"
    spec=importlib.util.spec_from_file_location("pinned_coconut_dataset",path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {path}")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independently_validate_raw(raw_path: Path, processed_path: Path) -> dict:
    # Coconut's pinned preprocessor defines one record per physical text-file
    # line via file.readlines(). str.splitlines() recognizes additional Unicode
    # and control-character separators and therefore can invent record
    # boundaries that the upstream preprocessor never uses.
    with raw_path.open("r", encoding="utf-8", newline=None) as handle:
        raw_lines=[line.rstrip("\r\n") for line in handle.readlines()]
    processed=json.loads(processed_path.read_text(encoding="utf-8"))
    if len(raw_lines) != len(processed):
        raise RuntimeError(
            f"record count mismatch {raw_path.name}: {len(raw_lines)} != {len(processed)}"
        )

    malformed=0
    for index,(line,item) in enumerate(zip(raw_lines,processed)):
        if "||" not in line or "##" not in line:
            malformed += 1
            continue
        # Recompute the pinned preprocessor semantics independently. It takes
        # the first question segment, the first CoT segment before "##", and
        # the final answer segment after the LAST "##" (some records contain
        # additional "##" markers inside the reasoning text).
        pipe_parts=line.split("||")
        hash_parts=line.split("##")
        if len(pipe_parts) < 2 or len(hash_parts) < 2:
            malformed += 1
            continue
        question=pipe_parts[0]
        expected_steps=pipe_parts[1].split("##")[0].strip().split(" ")
        expected_answer=hash_parts[-1].strip()

        if item["question"] != question:
            raise RuntimeError(f"question mismatch at {raw_path.name}:{index}")
        if item["steps"] != expected_steps:
            raise RuntimeError(f"steps mismatch at {raw_path.name}:{index}")
        if item["answer"] != expected_answer:
            raise RuntimeError(f"answer mismatch at {raw_path.name}:{index}")

    if malformed:
        raise RuntimeError(f"{raw_path.name}: {malformed} malformed raw records")

    broad_splitline_count=len(raw_path.read_text(encoding="utf-8").splitlines())

    return {
        "records":len(processed),
        "physical_line_records":len(raw_lines),
        "broad_splitline_count":broad_splitline_count,
        "extra_control_separator_boundaries":broad_splitline_count-len(raw_lines),
        "raw_sha256":sha256(raw_path),
        "processed_sha256":sha256(processed_path),
        "min_steps":min(len(item["steps"]) for item in processed),
        "max_steps":max(len(item["steps"]) for item in processed),
    }


def main() -> None:
    coconut_root=Path(os.environ["COCONUT_UPSTREAM"])
    data_root=Path(os.environ["COCONUT_DATA"])

    dataset_module=load_dataset_module(coconut_root)

    sources={}
    for split in ("train","valid","test"):
        sources[split]=independently_validate_raw(
            data_root/f"gsm_{split}.txt",
            data_root/f"gsm_{split}.json",
        )

    tokenizer=AutoTokenizer.from_pretrained(
        "openai-community/gpt2",
        revision=GPT2_REVISION,
    )
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id=tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id=tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id=tokenizer.convert_tokens_to_ids("<|end-latent|>")

    base_train=dataset_module.get_dataset(
        data_root/"gsm_train.json",
        tokenizer,
        max_size=8,
    )
    base_valid=dataset_module.get_dataset(
        data_root/"gsm_valid.json",
        tokenizer,
        max_size=8,
    )

    config=SimpleNamespace(
        pad_latent_to_max=True,
        max_latent_stage=3,
        c_thought=2,
        uniform_prob=0.0,
        no_cot=False,
    )

    stage=1
    latent_train=dataset_module.get_cot_latent_dataset(
        stage,
        base_train,
        config,
        start_id,
        latent_id,
        end_id,
        no_special_marker=False,
        shuffle=False,
    )
    latent_valid=dataset_module.get_question_latent_dataset(
        stage,
        base_valid,
        config,
        start_id,
        latent_id,
        end_id,
        no_special_marker=False,
    )

    train_features=[latent_train[i] for i in range(min(4,len(latent_train)))]
    collator=dataset_module.MyCollator(
        tokenizer,
        latent_id=latent_id,
        label_pad_token_id=-100,
    )
    batch=collator(train_features)

    expected_latent_tokens=stage*config.c_thought
    for index,feature in enumerate(train_features):
        actual=feature["input_ids"].count(latent_id)
        if actual != expected_latent_tokens:
            raise RuntimeError(
                f"feature {index}: expected {expected_latent_tokens} latent tokens, got {actual}"
            )
        if len(feature["input_ids"]) != len(feature["labels"]):
            raise RuntimeError(f"feature {index}: input/label length mismatch")

    if batch["input_ids"].shape != batch["labels"].shape:
        raise RuntimeError("collated input_ids/labels shape mismatch")
    if batch["input_ids"].shape != batch["attention_mask"].shape:
        raise RuntimeError("collated input_ids/attention_mask shape mismatch")
    if batch["input_ids"].shape != batch["position_ids"].shape:
        raise RuntimeError("collated input_ids/position_ids shape mismatch")

    ignored=int((batch["labels"] == -100).sum().item())
    latent_count=int((batch["input_ids"] == latent_id).sum().item())
    if ignored <= 0 or latent_count != expected_latent_tokens*len(train_features):
        raise RuntimeError("unexpected label masking or latent-token count")

    valid_features=[latent_valid[i] for i in range(min(2,len(latent_valid)))]
    valid_batch=collator(valid_features)
    if "labels" in valid_batch:
        raise RuntimeError("generation validation batch unexpectedly contains labels")

    payload={
        "schema_version":1,
        "evidence_class":"pinned-upstream-data-pipeline-preflight",
        "coconut":{
            "repository":"facebookresearch/coconut",
            "revision":COCONUT_REVISION,
            "dataset_module":"dataset.py",
            "preprocessor":"preprocessing/gsm_icot.py",
        },
        "data_source":{
            "repository":"da03/Internalize_CoT_Step_by_Step",
            "revision":DATA_REVISION,
            "license":"MIT",
            "underlying_dataset":"GSM8K",
            "underlying_dataset_license":"MIT",
        },
        "tokenizer":{
            "id":"openai-community/gpt2",
            "revision":GPT2_REVISION,
            "vocab_size_after_latent_tokens":len(tokenizer),
            "padding_side":tokenizer.padding_side,
        },
        "source_files":sources,
        "bounded_dataset":{
            "train_records_loaded":len(base_train),
            "valid_records_loaded":len(base_valid),
            "scheduled_stage":stage,
            "c_thought":config.c_thought,
            "latent_tokens_per_training_example":expected_latent_tokens,
        },
        "collator":{
            "training_batch_shape":list(batch["input_ids"].shape),
            "generation_batch_shape":list(valid_batch["input_ids"].shape),
            "latent_tokens_in_training_batch":latent_count,
            "ignored_training_labels":ignored,
            "has_position_ids":True,
        },
        "claim_boundary":(
            "This validates exact-source preprocessing, tokenization, scheduled "
            "latent-dataset construction, and collator mechanics on bounded CPU "
            "samples. It does not train a model or reproduce GSM8K accuracy."
        ),
    }
    print(json.dumps(payload,sort_keys=True))


if __name__=="__main__":
    main()
