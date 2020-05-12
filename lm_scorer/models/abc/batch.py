# pylint: disable=abstract-method
from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import abstractmethod

import torch

from .base import LMScorer


class BatchedLMScorer(LMScorer):
    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        batch_size = options.get("batch_size", 1)
        if batch_size < 1:
            raise ValueError("The batch_size option must be positive")
        # pylint: disable=attribute-defined-outside-init
        self.batch_size = batch_size

    # @overrides
    def _tokens_log_prob(
        self, text: List[str]
    ) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        outputs = []
        for i in range(0, len(text), self.batch_size):
            batch = text[i : i + self.batch_size]
            outputs.extend(self._tokens_log_prob_for_batch(batch))
        return outputs

    @abstractmethod
    def _tokens_log_prob_for_batch(
        self, text: List[str]
    ) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        ...  # pragma: no cover
