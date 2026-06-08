# pylint: disable=abstract-method
from .batch import BatchedLMScorer


class TransformersLMScorer(BatchedLMScorer):
    pass
