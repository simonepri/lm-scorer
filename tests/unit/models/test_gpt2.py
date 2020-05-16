# pylint: disable=missing-module-docstring,missing-function-docstring,unused-variable,too-many-locals,too-many-statements
import math
import pytest  # pylint: disable=unused-import

from lm_scorer.models.gpt2 import GPT2LMScorer


def describe_init():
    def should_throw_an_exception_for_an_unsupported_model_name():
        with pytest.raises(OSError):
            GPT2LMScorer("_")

    def should_not_throw_an_exception_for_a_supported_model_name():
        GPT2LMScorer("gpt2")


def describe_supported_model_names():
    def should_not_be_empty():
        assert len(list(GPT2LMScorer.supported_model_names())) > 0


def describe_tokens_log_prob_for_batch():
    # pylint: disable=protected-access
    scorer = GPT2LMScorer("gpt2")

    def should_work_on_zero_sentences():
        assert scorer._tokens_log_prob_for_batch([]) == []

    def should_work_on_an_empty_sentence():
        sentence = ""

        outputs = scorer._tokens_log_prob_for_batch([sentence])
        assert len(outputs) == 1

        scores, ids, tokens = outputs[0]
        assert len(scores) == 1
        assert len(ids) == 1
        assert len(tokens) == 1
        assert all(score <= 0.0 for score in scores)

    def should_work_on_a_single_sentence():
        sentences = ["Hello World!"]
        eps = 1e-4
        expected = [
            (
                [
                    -8.293977140323449,
                    -5.755700968654786,
                    -1.474859642902672,
                    -6.430916815662243,
                ],
                [15496, 2159, 0, 50256],
                ["Hello", "ĠWorld", "!", "<|endoftext|>"],
            )
        ]
        outputs = scorer._tokens_log_prob_for_batch(sentences)
        assert len(outputs) == len(sentences)

        for i in range(len(sentences)):
            scores, ids, tokens = outputs[i]
            scores, ids = scores.tolist(), ids.tolist()
            exp_scores, exp_ids, exp_tokens = expected[i]
            for j, (score, exp_score) in enumerate(zip(scores, exp_scores)):
                assert math.isclose(score, exp_score, rel_tol=eps), {"index": (i, j)}
            assert ids == exp_ids
            assert tokens == exp_tokens

    def should_work_on_multiple_sentences():
        sentences = ["This is an example.", "The sky is blue today."]
        eps = 1e-4
        expected = [
            (
                [
                    -4.858048795596886,
                    -1.6949359403515274,
                    -2.42076125597049,
                    -3.642987703024289,
                    -6.384910922168785,
                    -6.194488923833262,
                ],
                [1212, 318, 281, 1672, 13, 50256],
                ["This", "Ġis", "Ġan", "Ġexample", ".", "<|endoftext|>"],
            ),
            (
                [
                    -3.2781003703039175,
                    -8.624397355117694,
                    -1.1694785129190564,
                    -2.719566557397684,
                    -4.5889773146928405,
                    -1.297036181310574,
                    -5.455337295354369,
                ],
                [464, 6766, 318, 4171, 1909, 13, 50256],
                ["The", "Ġsky", "Ġis", "Ġblue", "Ġtoday", ".", "<|endoftext|>"],
            ),
        ]
        outputs = scorer._tokens_log_prob_for_batch(sentences)
        assert len(outputs) == len(sentences)

        for i in range(len(sentences)):
            scores, ids, tokens = outputs[i]
            scores, ids = scores.tolist(), ids.tolist()
            exp_scores, exp_ids, exp_tokens = expected[i]
            for j, (score, exp_score) in enumerate(zip(scores, exp_scores)):
                assert math.isclose(score, exp_score, rel_tol=eps), {"index": (i, j)}
            assert ids == exp_ids
            assert tokens == exp_tokens

    @pytest.mark.xfail
    def should_work_with_a_sentence_longer_than_the_model_max_size():
        max_input_size = scorer.tokenizer.max_len
        long_sentence = "Very" + " long" * max_input_size
        exp_ids = [16371] + [809] * max_input_size + [50256]
        exp_tokens = ["Very"] + ["Ġlong"] * max_input_size + ["<|endoftext|>"]

        _, ids, tokens = scorer._tokens_log_prob_for_batch([long_sentence])[0]
        ids = ids.tolist()

        assert ids == exp_ids
        assert tokens == exp_tokens
