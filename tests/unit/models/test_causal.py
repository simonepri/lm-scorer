import math

from lm_scorer.models.causal import CausalLMScorer


def describe_tokens_log_prob_for_batch():
    def should_match_the_gpt2_scorer_on_gpt2():
        # The generic scorer must reproduce the GPT-2 scores byte-for-byte so
        # that existing (and published) results are unchanged.
        scorer = CausalLMScorer("gpt2")
        scores, _, tokens = scorer._tokens_log_prob_for_batch(["Hello World!"])[0]
        expected = [
            -8.293975830078125,
            -5.755699157714844,
            -1.4748573303222656,
            -6.430915832519531,
        ]
        assert tokens == ["Hello", "ĠWorld", "!", "<|endoftext|>"]
        for score, exp in zip(scores.tolist(), expected):
            assert math.isclose(score, exp, rel_tol=1e-4)


def describe_other_causal_models():
    def should_score_a_non_gpt2_model():
        # A non-GPT-2 architecture (GPTNeoX) with no pad token must work without
        # resizing the embeddings, and rank a grammatical sentence higher.
        scorer = CausalLMScorer("EleutherAI/pythia-70m", batch_size=2)
        good = scorer.sentence_score("The cat sat on the mat.", log=True)
        bad = scorer.sentence_score("Cat the mat on sat the.", log=True)
        assert good > bad

    def should_score_when_special_tokens_are_beyond_the_model_vocab():
        # dbmdz/german-gpt2 ships an eos token id beyond its trained embeddings;
        # the scorer must skip those untrained bos/eos tokens (rather than crash
        # or pollute the scores) and still rank a grammatical sentence higher.
        scorer = CausalLMScorer("dbmdz/german-gpt2", batch_size=2)
        good = scorer.sentence_score("Der Hund läuft im Park.", log=True)
        bad = scorer.sentence_score("Park im läuft Hund der.", log=True)
        assert good > bad
