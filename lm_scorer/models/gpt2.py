from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from .abc.transformers import TransformersLMScorer


class GPT2LMScorer(TransformersLMScorer):
    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        # pylint: disable=attribute-defined-outside-init
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        if "device" in options:
            self.model.to(options["device"])

    # @overrides
    def _tokens_log_prob_single_sentence(
        self, text: str
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, List[str]]:
        device = self.model.device
        input_text = "%s%s%s" % (
            self.tokenizer.bos_token,
            text,
            self.tokenizer.eos_token,
        )
        # len(tokens) = seq_len + 2
        tokens = self.tokenizer.tokenize(input_text)
        # ids.shape = [1, seq_len + 2]
        ids = torch.tensor(  # pylint: disable=not-callable
            [self.tokenizer.convert_tokens_to_ids(tokens)],
            device=device,
            dtype=torch.long,
        )

        with torch.no_grad():
            outputs = self.model(ids)

        # pred_scores.shape = [1, seq_len + 2, vocab_size]
        pred_scores = outputs[0]

        # len(tokens) = seq_len + 1
        tokens = tokens[1:]
        # ids.shape = [1, seq_len + 1, vocab_size]
        ids = ids[:, 1:]
        # pred_scores.shape = [1, seq_len + 1, vocab_size]
        pred_scores = pred_scores[:, :-1, :]

        # ids_scores.shape = [1, seq_len + 1]
        ids_scores = pred_scores.gather(2, ids.unsqueeze(2)).squeeze(2)
        # log_prob.shape = [1, seq_len + 1]
        log_probs = ids_scores - pred_scores.logsumexp(2)

        return log_probs[0], ids[0], tokens  # type: ignore

    def _tokens_log_prob(
        self, sentences: List[str]
    ) -> List[Tuple[torch.FloatTensor, torch.LongTensor, List[str]]]:
        return [
            self._tokens_log_prob_single_sentence(sentence) for sentence in sentences
        ]

    # @overrides
    @classmethod
    def _supported_model_names(cls) -> Iterable[str]:
        return GPT2LMHeadModel.pretrained_model_archive_map.keys()
