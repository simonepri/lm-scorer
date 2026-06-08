import pytest

from lm_scorer.models.auto import AutoLMScorer
from lm_scorer.models.causal import CausalLMScorer
from lm_scorer.models.gpt2 import GPT2LMScorer


def describe_init():
    def should_throw_an_exception_when_called():
        with pytest.raises(EnvironmentError):
            AutoLMScorer()


def describe_supported_model_names():
    def should_not_be_empty():
        assert len(list(AutoLMScorer.supported_model_names())) > 0


def describe_from_pretrained():
    def should_return_gpt2_models(mocker):
        mocker.patch.object(GPT2LMScorer, "__init__", return_value=None)
        for model_name in GPT2LMScorer.supported_model_names():
            scorer = AutoLMScorer.from_pretrained(model_name)
            assert isinstance(scorer, GPT2LMScorer), model_name

    def should_dispatch_other_models_to_the_causal_scorer(mocker):
        mocker.patch.object(CausalLMScorer, "__init__", return_value=None)
        scorer = AutoLMScorer.from_pretrained("bigscience/bloom-560m")
        assert isinstance(scorer, CausalLMScorer)
