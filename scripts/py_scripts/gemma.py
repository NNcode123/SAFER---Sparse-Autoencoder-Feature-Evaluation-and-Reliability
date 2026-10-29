from functools import lru_cache
from typing import TypedDict

import torch
from sae_lens import SAE
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class GemmaModelConfig(TypedDict):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    gemma_encoders: list[SAE]


model_config_dct: dict[str, GemmaModelConfig] = {}


def load_gemma_resources() -> dict[str, GemmaModelConfig]:
    model_id = "google/gemma-3-4b-pt"
    print("Loading Gemma model and Gemma Scope SAEs")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    gemma = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    auto_encoders = [
        SAE.from_pretrained(
            release="gemma-scope-2-4b-pt-res-all",
            sae_id=f"layer_{index}_width_16k_l0_big",
            device="cpu",
        )
        for index, _ in enumerate(gemma.model.language_model.layers)
    ]

    model_config_dct["gemma"] = {
        "model": gemma,
        "tokenizer": tokenizer,
        "gemma_encoders": auto_encoders,
    }

    return model_config_dct
