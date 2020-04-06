# pylint: disable=abstract-method
from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import abstractmethod

import os

from .base import LMScorer


class TransformersLMScorer(LMScorer):
    # @overrides
    @abstractmethod
    def build(self, model_name: str, options: Dict[str, str]) -> None:
        super().build(model_name, options)

        #  Make transformers cache path configurable.
        cache_dir = os.environ.get("TRANSFORMERS_CACHE_DIR", ".transformers_cache")
        options["cache_dir"] = options.get("cache_dir", cache_dir)
