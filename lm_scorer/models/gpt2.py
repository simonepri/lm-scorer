from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from .abc.transformers import TransformersLMScorer


class GPT2LMScorer(TransformersLMScorer):
    # @overrides
    def build(self, model_name: str, options: Dict[str, str]) -> None:
        super().build(model_name, options)

        # pylint: disable=attribute-defined-outside-init
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, **options)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, **options)
        self.model.eval()

    # @overrides
    def score(
        self, text: str, return_log_prob: bool = True, return_tokens: bool = False
    ) -> Union[float, Tuple[float, Dict[str, float]]]:
        input_text = "%s%s%s" % (
            self.tokenizer.bos_token,
            text,
            self.tokenizer.eos_token,
        )
        tokens = self.tokenizer.tokenize(input_text)
        # ids.shape = [1, seq_len + 2]
        ids = torch.LongTensor(
            [self.tokenizer.convert_tokens_to_ids(tokens)]
        )  # type: torch.Tensor # type: ignore

        with torch.no_grad():
            outputs = self.model(ids)

        # pred_scores.shape = [1, seq_len + 2, vocab_size]
        pred_scores = outputs[0]

        # ids.shape = [1, seq_len + 1, vocab_size]
        ids = ids[:, 1:]
        # pred_scores.shape = [1, seq_len + 1, vocab_size]
        pred_scores = pred_scores[:, :-1, :]

        # ids_scores.shape = [1, seq_len + 1]
        ids_scores = pred_scores.gather(2, ids.unsqueeze(2)).squeeze(2)
        # log_prob.shape = [1, seq_len + 1]
        log_probs = ids_scores - pred_scores.logsumexp(2)

        # scores.shape = [1]
        scores = log_probs.sum(1) if return_log_prob else log_probs.sum(1).exp()

        ret = scores[0].item()

        if return_tokens:
            # token_scores.shape = [1, seq_len + 1]
            token_scores = log_probs if return_log_prob else log_probs.exp()
            ret = (ret, dict(zip(tokens[1:], token_scores[0].tolist())))

        return ret

    # @overrides
    @classmethod
    def supported_model_names(cls) -> Iterable[str]:
        return GPT2LMHeadModel.pretrained_model_archive_map.keys()
