#!/usr/bin/env python3
import copy
import unittest

from tools.validate_coconut_stage0_plan import (
    ALLOWED_OVERRIDE_KEYS,
    effective_global_batch,
    validate_plan,
)


REFERENCE={
    "project":"coconut",
    "save_path":"YOUR_PATH_TO_SAVE_THE_MODEL",
    "name":"gsm-cot",
    "only_eval":False,
    "coconut":False,
    "cot":True,
    "no_thoughts":False,
    "no_cot":False,
    "c_thought":0,
    "epochs_per_stage":1,
    "max_latent_stage":0,
    "pad_latent_to_max":True,
    "save_only_improve":True,
    "uniform_prob":0.0,
    "model_id":"openai-community/gpt2",
    "load_model_path":"None",
    "seed":0,
    "resume":0,
    "bf16":False,
    "train_path":"data/gsm_train.json",
    "val_path":"data/gsm_valid.json",
    "reset_optimizer":False,
    "batch_size_training":32,
    "debug":False,
    "gradient_accumulation_steps":1,
    "num_epochs":25,
    "lr":1e-4,
    "weight_decay":0.01,
}

OVERRIDES={
    "save_path":"/out/checkpoints",
    "name":"gsm-cot-stage0-2gpu",
    "model_id":"/opt/models/gpt2",
    "train_path":"/opt/coconut/data/gsm_train.json",
    "val_path":"/opt/coconut/data/gsm_valid.json",
    "gradient_accumulation_steps":2,
}

PLAN={
    "id":"campaign-0001/coconut-gsm-cot-stage0-2gpu",
    "reference_execution":{
        "world_size":4,
        "per_rank_batch_size":32,
        "gradient_accumulation_steps":1,
        "effective_global_batch":128,
    },
    "target_execution":{
        "world_size":2,
        "per_rank_batch_size":32,
        "gradient_accumulation_steps":2,
        "effective_global_batch":128,
        "launcher":"torchrun",
        "wandb_mode":"offline",
    },
    "allowed_config_overrides":OVERRIDES,
    "immutable_inputs":{},
}


class Stage0PlanTests(unittest.TestCase):
    def derived(self):
        d=copy.deepcopy(REFERENCE)
        d.update(OVERRIDES)
        return d

    def test_effective_global_batch(self):
        self.assertEqual(effective_global_batch(4,32,1),128)
        self.assertEqual(effective_global_batch(2,32,2),128)

    def test_exact_allowed_differences_pass(self):
        r=validate_plan(plan=PLAN,upstream_config=REFERENCE,derived_config=self.derived())
        self.assertEqual(r["reference_effective_global_batch"],128)
        self.assertEqual(r["target_effective_global_batch"],128)
        self.assertEqual(set(r["allowed_config_differences"]),ALLOWED_OVERRIDE_KEYS)
        self.assertFalse(r["launcher_uses_shell"])

    def test_scientific_hyperparameter_change_fails(self):
        d=self.derived()
        d["lr"]=2e-4
        with self.assertRaises(ValueError):
            validate_plan(plan=PLAN,upstream_config=REFERENCE,derived_config=d)

    def test_global_batch_change_fails(self):
        d=self.derived()
        d["gradient_accumulation_steps"]=1
        plan=copy.deepcopy(PLAN)
        plan["allowed_config_overrides"]["gradient_accumulation_steps"]=1
        with self.assertRaises(ValueError):
            validate_plan(plan=plan,upstream_config=REFERENCE,derived_config=d)


if __name__=="__main__":
    unittest.main()
