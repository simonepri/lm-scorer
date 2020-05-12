from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import ABC, abstractmethod

import math

import torch


class LMScorer(ABC):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self._build(model_name, kwargs)

    @overload
    def sentence_score(
        self, text: str, log: bool = False, reduce: str = "prod"
    ) -> float:
        ...

    @overload
    def sentence_score(
        self, text: List[str], log: bool = False, reduce: str = "prod"
    ) -> List[float]:
        ...

    def sentence_score(
        self, text: Union[str, List[str]], log: bool = False, reduce: str = "prod",
    ) -> Union[float, List[float]]:
        sentences = [text] if isinstance(text, str) else text
        scores: List[float] = []
        if len(sentences) == 0:
            return scores

        outputs = self._tokens_log_prob(sentences)
        for output in outputs:
            log_probs = output[0]
            tlen = log_probs.shape[0]

            if reduce == "prod":
                score = log_probs.sum()
            elif reduce == "mean":
                score = log_probs.logsumexp(0) - math.log(tlen)
            elif reduce == "gmean":
                score = log_probs.mean(0)
            elif reduce == "hmean":
                score = log_probs.neg().logsumexp(0).neg() + math.log(tlen)
            else:
                raise ValueError("Unrecognized scoring strategy: %s" % reduce)
            if not log:
                score = score.exp()

            scores.append(score.item())

        return scores[0] if isinstance(text, str) else scores

    @overload
    def tokens_score(
        self, text: str, log: bool = False
    ) -> Tuple[List[float], List[int], List[str]]:
        ...

    @overload
    def tokens_score(
        self, text: List[str], log: bool = False
    ) -> List[Tuple[List[float], List[int], List[str]]]:
        ...

    def tokens_score(
        self, text: Union[str, List[str]], log: bool = False
    ) -> Union[
        Tuple[List[float], List[int], List[str]],
        List[Tuple[List[float], List[int], List[str]]],
    ]:
        sentences = [text] if isinstance(text, str) else text
        outputs: List[Tuple[List[float], List[int], List[str]]] = []
        if len(sentences) == 0:
            return outputs

        for log_probs, ids, tokens in self._tokens_log_prob(sentences):
            scores = log_probs if log else log_probs.exp()
            scores = cast(torch.DoubleTensor, scores)
            output = (scores.tolist(), ids.tolist(), tokens)
            outputs.append(output)

        return outputs[0] if isinstance(text, str) else outputs

    @classmethod
    def supported_model_names(cls) -> Iterable[str]:
        return cls._supported_model_names()

    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        # pylint: disable=attribute-defined-outside-init, unused-argument
        self.model_name = model_name

    @abstractmethod
    def _tokens_log_prob(
        self, text: List[str]
    ) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        ...  # pragma: no cover

    @classmethod
    @abstractmethod
    def _supported_model_names(cls) -> Iterable[str]:
        ...  # pragma: no cover
