from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from .abc.transformers import TransformersLMScorer


class GPT2LMScorer(TransformersLMScorer):
    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        # pylint: disable=attribute-defined-outside-init
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, add_special_tokens=False
        )
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        if "device" in options:
            self.model.to(options["device"])

    def _add_special_tokens(self, text: str) -> str:
        return self.tokenizer.bos_token + text + self.tokenizer.eos_token

    # @overrides
    def _tokens_log_prob_for_batch(
        self, text: List[str]
    ) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        device = self.model.device

        outputs = []
        for sentence in text:
            # len(sentence) = seq_len + 2
            sentence = self._add_special_tokens(sentence)
            # len(tokens) = seq_len + 2
            tokens: List[str] = self.tokenizer.tokenize(sentence)
            # ids.shape = [1, seq_len + 2]
            ids = torch.tensor(  # pylint: disable=not-callable
                [self.tokenizer.convert_tokens_to_ids(tokens)],
                device=device,
                dtype=torch.long,
            )

            with torch.no_grad():
                model_outputs: Tuple[torch.Tensor] = self.model(ids)

            # pred_scores.shape = [1, seq_len + 2, vocab_size]
            pred_scores = model_outputs[0].double()

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

            log_probs = log_probs.squeeze(0)
            ids = ids.squeeze(0)

            log_probs = cast(torch.DoubleTensor, log_probs)
            ids = cast(torch.LongTensor, ids)

            output = (log_probs, ids, tokens)
            outputs.append(output)
        return outputs

    # @overrides
    @classmethod
    def _supported_model_names(cls) -> Iterable[str]:
        return GPT2LMHeadModel.pretrained_model_archive_map.keys()
