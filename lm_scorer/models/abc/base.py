from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import ABC, abstractmethod


class LMScorer(ABC):
    def __init__(self, model_name: str, **kwargs: str) -> None:
        self.build(model_name, kwargs)

    @abstractmethod
    def build(self, model_name: str, options: Dict[str, str]) -> None:
        ...  # pragma: no cover

    @abstractmethod
    def score(
        self, text: str, return_log_prob: bool = True, return_tokens: bool = False
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        ...  # pragma: no cover

    @classmethod
    @abstractmethod
    def supported_model_names(cls) -> Iterable[str]:
        ...  # pragma: no cover
