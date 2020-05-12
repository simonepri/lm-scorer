# pylint: disable=missing-module-docstring,missing-function-docstring,unused-variable,too-many-locals,too-many-statements
import pytest  # pylint: disable=unused-import

from lm_scorer.models.auto import AutoLMScorer
from lm_scorer.models.gpt2 import GPT2LMScorer


def describe_init():
    def should_throw_an_exception_when_called():
        with pytest.raises(EnvironmentError):
            AutoLMScorer()


def describe_supported_model_names():
    def should_not_be_empty():
        assert len(list(AutoLMScorer.supported_model_names())) > 0


def describe_from_pretrained():
    def should_throw_an_error_for_an_unsupported_model_name():
        with pytest.raises(ValueError):
            AutoLMScorer.from_pretrained("_")

    def should_return_gpt2_models(mocker):
        mocker.patch.object(GPT2LMScorer, "__init__", return_value=None)
        for model_name in GPT2LMScorer.supported_model_names():
            scorer = AutoLMScorer.from_pretrained(model_name)
            assert isinstance(scorer, GPT2LMScorer), model_name
