# pylint: disable=missing-module-docstring,missing-function-docstring,unused-variable,too-many-locals,too-many-statements
import math
import pytest  # pylint: disable=unused-import

import scipy
import torch
from lm_scorer.models.abc.base import LMScorer


def model(text):
    tokens = ["START"] + text.split(" ")
    scores = [float(-i) for i in range(len(tokens))]
    ids = list(range(len(tokens)))
    return scores, ids, tokens


class FixtureLMScorer(LMScorer):
    def _tokens_log_prob(self, text):
        outputs = []
        for sentence in text:
            scores, ids, tokens = model(sentence)
            # pylint: disable=not-callable
            scores = torch.tensor(scores)
            # pylint: disable=not-callable
            ids = torch.tensor(ids)
            output = (scores, ids, tokens)
            outputs.append(output)
        return outputs

    @classmethod
    def _supported_model_names(cls):
        return iter([])

    def fix_sentence_score(self, text, reduce="prod", log=False):
        sentences = [text] if isinstance(text, str) else text
        scores = []
        if len(sentences) == 0:
            return scores

        outputs = self._tokens_log_prob(sentences)
        for output in outputs:
            log_probs = output[0]
            probs = log_probs.exp()

            if reduce == "prod":
                score = probs.prod()
            elif reduce == "mean":
                score = probs.mean()
            elif reduce == "gmean":
                # pylint: disable=not-callable
                score = torch.tensor(scipy.stats.gmean(probs.numpy()))
            elif reduce == "hmean":
                # pylint: disable=not-callable
                score = torch.tensor(scipy.stats.hmean(probs.numpy()))
            else:
                raise ValueError("Unrecognized scoring strategy: %s" % reduce)
            if log:
                score = score.log()

            scores.append(score.item())

        return scores[0] if isinstance(text, str) else scores

    def fix_tokens_score(self, text, log=False):
        sentences = [text] if isinstance(text, str) else text
        outputs = []
        if len(sentences) == 0:
            return outputs

        for log_probs, ids, tokens in self._tokens_log_prob(sentences):
            scores = log_probs if log else log_probs.exp()
            output = (scores.tolist(), ids.tolist(), tokens)
            outputs.append(output)

        return outputs[0] if isinstance(text, str) else outputs


def describe_sentence_score():
    scorer = FixtureLMScorer("")

    def should_return_the_correct_number_of_results():
        scores = scorer.sentence_score([])
        assert isinstance(scores, list) and len(scores) == 0
        scores = scorer.sentence_score(["A"])
        assert isinstance(scores, list) and len(scores) == 1
        scores = scorer.sentence_score(["A", "B"])
        assert isinstance(scores, list) and len(scores) == 2
        scores = scorer.sentence_score("A")
        assert isinstance(scores, float)

    def should_correctly_compute_every_reduction_strategy():
        text = "Hello World"
        eps = 1e-6
        strategies = ["prod", "mean", "gmean", "hmean"]
        for strategy in strategies:
            for log in [False, True]:
                context = (strategy, log)

                score = scorer.sentence_score("", reduce=strategy, log=log)
                expected = scorer.fix_sentence_score("", reduce=strategy, log=log)
                assert isinstance(score, float)
                assert math.isclose(score, expected, rel_tol=eps), context

                score = scorer.sentence_score(text, reduce=strategy, log=log)
                expected = scorer.fix_sentence_score(text, reduce=strategy, log=log)
                context = ((strategy, log), (score, expected))
                assert isinstance(score, float)
                assert math.isclose(score, expected, rel_tol=eps), context


def describe_tokens_score():
    scorer = FixtureLMScorer("")

    def should_return_the_correct_number_of_results():
        info = scorer.tokens_score([])
        assert isinstance(info, list) and len(info) == 0
        info = scorer.tokens_score(["A"])
        assert isinstance(info, list) and len(info) == 1
        info = scorer.tokens_score(["A", "B"])
        assert isinstance(info, list) and len(info) == 2
        info = scorer.tokens_score("A")
        assert isinstance(info, tuple) and len(info) == 3
