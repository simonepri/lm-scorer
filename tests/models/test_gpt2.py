# pylint: disable=missing-module-docstring,missing-function-docstring,unused-variable,too-many-locals,too-many-statements
import pytest  # pylint: disable=unused-import

from lm_scorer.models.gpt2 import GPT2LMScorer


def assert_score_of_sentence_pairs(scorer, sentence_pairs):
    errors = []
    for i, sentence_pair in enumerate(sentence_pairs):
        correct_sentence, wrong_sentence = sentence_pair
        correct_score = scorer.score(correct_sentence)
        wrong_score = scorer.score(wrong_sentence)
        if not correct_score > wrong_score:
            errors.append(i)
    assert errors == []


def describe_init():
    def should_throw_an_exception_for_an_unsupported_model_name():
        with pytest.raises(OSError):
            GPT2LMScorer("_")

    def should_not_throw_an_exception_for_a_supported_model_name():
        GPT2LMScorer("gpt2")


def describe_supported_model_names():
    def should_not_be_empty():
        assert len(list(GPT2LMScorer.supported_model_names())) > 0


def describe_score_for_english():
    scorer = GPT2LMScorer("gpt2")

    def should_work_on_an_empty_sentence():
        score_1 = scorer.score("", return_log_prob=False)
        assert 0.0 <= score_1 <= 1.0
        score_2, token_scores = scorer.score(
            "", return_log_prob=False, return_tokens=True
        )
        assert score_1 == score_2
        assert token_scores == {"<|endoftext|>": score_2}

        score_1 = scorer.score("", return_log_prob=True)
        assert score_1 <= 0.0
        score_2, token_scores = scorer.score(
            "", return_tokens=True, return_log_prob=True
        )
        assert score_1 == score_2
        assert token_scores == {"<|endoftext|>": score_2}

    def should_give_lower_score_to_sentences_with_adjectives_errors():
        # ERRANT - ADJ error
        sentence_pairs = [
            ("I have a big amount of money.", "I have a wide amount of money."),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_adjectives_forms_errors():
        # ERRANT - ADJ:FORM error
        sentence_pairs = [
            (
                "This is the best day of my life.",
                "This is the goodest day of my life.",
            ),
            (
                "I think that is the biggest apple I have ever seen!",
                "I think that is the bigger apple I have ever seen!",
            ),
            (
                "I think this game is easier than the one we played yesterday.",
                "I think this game is more easy than the one we played yesterday.",
            ),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_adverbs_errors():
        # ERRANT - ADV error
        sentence_pairs = [
            ("Let us finish this quickly!", "Let us finish this speedily!",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_conjunctions_errors():
        # ERRANT - CONJ error
        sentence_pairs = [
            (
                "It is a private matter between him and me.",
                "It is a private matter between him but me.",
            ),
            (
                "I need to finish this project, but I do not have enough time.",
                "I need to finish this project, and I do not have enough time.",
            ),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_determiners_errors():
        # ERRANT - DET error
        sentence_pairs = [
            (
                "The restaurant is in the middle of my home town.",
                "The restaurant is in the middle of the my home town.",
            ),
            ("I am Italian.", "I am a Italian.",),
            ("I am a teacher.", "I am teacher.",),
            ("She gave me some advice.", "She gave me an advice.",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_morphology_errors():
        # ERRANT - MORPH error
        sentence_pairs = [
            ("I will quickly solve this.", "I will quick solve this.",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_noun_errors():
        # ERRANT - NOUN error
        sentence_pairs = [
            (
                "Today's people have a frenetic lifestyle.",
                "Today's person have a frenetic lifestyle.",
            ),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_noun_inflection_errors():
        # ERRANT - NOUN:INFL error
        sentence_pairs = [
            (
                "There is too many information in this table.",
                "There is too many informations in this table.",
            ),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_noun_number_errors():
        # ERRANT - NOUN:NUM error
        sentence_pairs = [
            ("She has too many cats.", "She has too many cat.",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_noun_possissive_errors():
        # ERRANT - NOUN:POSS error
        sentence_pairs = [
            ("My friend's boss is leaving.", "My friends boss is leaving.",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_orthography_errors():
        # ERRANT - ORTH error
        sentence_pairs = [
            ("You are my best friend.", "You are my bestfriend.",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_particle_errors():
        # ERRANT - PART error
        sentence_pairs = [
            ("Can you look at the kids?", "Can you look in the kids?",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_preposition_errors():
        # ERRANT - PREP error
        sentence_pairs = [
            ("Can you look at them?", "Can you look in them?",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_pronoun_errors():
        # ERRANT - PRON error
        sentence_pairs = [
            (
                "We should believe more in ourselves.",
                "We should believe more in ours.",
            ),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_punctuation_errors():
        # ERRANT - PUNCT error
        sentence_pairs = [
            ("I like dogs, cats, and dolphins.", "I like dogs cats and dolphins.",),
            ("I can do this, but not now.", "I can do this but not now.",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_spelling_errors():
        # ERRANT - SPELL error
        sentence_pairs = [
            (
                "They believe it is a genetic problem.",
                "They believe it is a genectic problem.",
            ),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_verb_errors():
        # ERRANT - VERB error
        sentence_pairs = [
            ("I can walk there.", "I can ambulate there.",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_verb_form_errors():
        # ERRANT - VERB:FORM error
        sentence_pairs = [
            ("I danced yesterday.", "I dancing yesterday.",),
            ("I am going to run a marathon.", "I am go to run a marathon.",),
            ("I am going to run a marathon.", "I am going to running a marathon.",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_verb_inflection_errors():
        # ERRANT - VERB:INFL error
        sentence_pairs = [
            ("I got arrested yesterday.", "I getted arrested yesterday.",),
            ("You flipped the wrong coin.", "You fliped the wrong coin.",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_verb_subj_agreement_errors():
        # ERRANT - VERB:SVA error
        sentence_pairs = [
            ("I think he has the virus.", "I think he have the virus.",),
            ("They said he is sick.", "They said he are sick.",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)

    def should_give_lower_score_to_sentences_with_verb_tense_errors():
        # ERRANT - VERB:TENSE error
        sentence_pairs = [
            ("He ate the pie yesterday.", "He eats the pie yesterday.",),
            ("The pie was eaten by him yesterday.", "The pie eats by him yesterday.",),
        ]
        assert_score_of_sentence_pairs(scorer, sentence_pairs)
