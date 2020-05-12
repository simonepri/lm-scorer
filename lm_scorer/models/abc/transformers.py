# pylint: disable=abstract-method
from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import os

from .batch import BatchedLMScorer


class TransformersLMScorer(BatchedLMScorer):
    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        #  Make transformers cache path configurable.
        cache_dir = os.environ.get("TRANSFORMERS_CACHE_DIR", ".transformers_cache")
        options["cache_dir"] = options.get("cache_dir", cache_dir)
