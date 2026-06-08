import itertools
from typing import Any, Iterable

from .abc.base import LMScorer
from .causal import CausalLMScorer
from .gpt2 import GPT2LMScorer


class AutoLMScorer:
    MODEL_CLASSES = [GPT2LMScorer]
    # Any model name that isn't matched above is loaded as a generic causal LM.
    FALLBACK_CLASS = CausalLMScorer

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
        classes = cls.MODEL_CLASSES
        models = map(lambda c: c.supported_model_names(), classes)
        return itertools.chain.from_iterable(models)
