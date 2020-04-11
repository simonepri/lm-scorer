from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import ABC, abstractmethod

import torch


class LMScorer(ABC):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._build(model_name, kwargs)

    def sentence_score(self, text: str, log: bool = False) -> float:
        log_probs, _, _ = self._tokens_log_prob(text)

        score = log_probs.sum()

        if not log:
            score = score.exp()

        return score.item()

    def tokens_score(
        self, text: str, log: bool = False
    ) -> Tuple[List[float], List[int], List[str]]:
        log_probs, ids, tokens = self._tokens_log_prob(text)

        scores = log_probs  # type: torch.Tensor # type: ignore
        if not log:
            scores = scores.exp()

        return scores.tolist(), ids.tolist(), tokens

    @classmethod
    def supported_model_names(cls) -> Iterable[str]:
        return cls._supported_model_names()

    @abstractmethod
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        ...  # pragma: no cover

    @abstractmethod
    def _tokens_log_prob(
        self, text: str
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, List[str]]:
        ...  # pragma: no cover

    @classmethod
    @abstractmethod
    def _supported_model_names(cls) -> Iterable[str]:
        ...  # pragma: no cover
