from typing import Any, Iterable

from .abc.base import LMScorer
from .causal import CausalLMScorer
from .gpt2 import GPT2LMScorer


class AutoLMScorer:
    MODEL_CLASSES = [GPT2LMScorer]
    # Any model name that isn't matched above is loaded as a generic causal LM.
    FALLBACK_CLASS = CausalLMScorer
    # A curated, tested selection of causal LMs surfaced to users. Any other
    # Hugging Face causal LM works too -- these are just well-known, ungated
    # starting points spanning languages, families and sizes.
    SUPPORTED_MODEL_NAMES = [
        # GPT-2 family
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "distilgpt2",
        # Multilingual GPT-2
        "ai-forever/mGPT",
        "dbmdz/german-gpt2",
        # EleutherAI
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/gpt-neo-125m",
        "EleutherAI/gpt-neo-1.3B",
        # Meta
        "facebook/opt-125m",
        "facebook/opt-1.3b",
        "facebook/xglm-564M",
        # BigScience (multilingual)
        "bigscience/bloom-560m",
        "bigscience/bloomz-560m",
        # Modern small
        "cerebras/Cerebras-GPT-111M",
        "HuggingFaceTB/SmolLM-135M",
        "HuggingFaceTB/SmolLM-360M",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
    ]

    def __init__(self):
        raise EnvironmentError(
            "AutoLMscorer is designed to be instantiated "
            "using the `AutoLMscorer.from_pretrained(model_name)`"
            "method"
        )

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: Any) -> LMScorer:
        for model_class in cls.MODEL_CLASSES:
            if model_name not in model_class.supported_model_names():
                continue
            return model_class(model_name, **kwargs)
        # Any other model is loaded as a generic causal language model.
        return cls.FALLBACK_CLASS(model_name, **kwargs)

    @classmethod
    def supported_model_names(cls) -> Iterable[str]:
        return cls.SUPPORTED_MODEL_NAMES
