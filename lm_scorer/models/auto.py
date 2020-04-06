from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import itertools

from .abc.base import LMScorer
from .gpt2 import GPT2LMScorer


class AutoLMScorer:
    MODEL_CLASSES = [GPT2LMScorer]

    def __init__(self):
        raise EnvironmentError(
            "AutoLMscorer is designed to be instantiated "
            "using the `AutoLMscorer.from_pretrained(model_name)`"
            "method"
        )

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs: str) -> LMScorer:
        for model_class in cls.MODEL_CLASSES:
            if model_name not in model_class.supported_model_names():
                continue
            return model_class(model_name, **kwargs)
        raise ValueError(
            "Unrecognized model name."
            "Can be one of: %s" % ", ".join(cls.supported_model_names()),
        )

    @classmethod
    def supported_model_names(cls) -> Iterable[str]:
        classes = cls.MODEL_CLASSES
        models = map(lambda c: c.supported_model_names(), classes)
        return itertools.chain.from_iterable(models)
