# pylint: disable=missing-module-docstring,missing-function-docstring,unused-variable,too-many-locals,too-many-statements
import pytest  # pylint: disable=unused-import

from lm_scorer.models.abc.batch import BatchedLMScorer


class FixtureLMScorer(BatchedLMScorer):
    def _tokens_log_prob_for_batch(self, text):
        return []

    @classmethod
    def _supported_model_names(cls):
        return iter([""])


def describe_batch_size_option():
    def should_throw_an_exception_if_non_positive():
        with pytest.raises(ValueError):
            FixtureLMScorer("", batch_size=0)
        with pytest.raises(ValueError):
            FixtureLMScorer("", batch_size=-1)

    def should_not_throw_an_exception_if_positive():
        FixtureLMScorer("", batch_size=1)
