from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import torch
import copy
from transformers import BertForMaskedLM, BertTokenizer

from .abc.transformers import TransformersLMScorer


class BERTLMScorer(TransformersLMScorer):
    """
    Use BERT to score a sentence following the idea describe in the paper
    Effective Sentence Scoring Method Using BERT for Speech Recognition. J. Shin, Y. Lee, Kyomin Jung

    Roughly the idea is to mask successively each token in the sentences and compute the log prob of true token
    """
    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        # pylint: disable=attribute-defined-outside-init
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        if "device" in options:
            self.model.to(options["device"])

    def _generate_mask_sentences(self, tokens: List[str]) -> List[List[str]]:
        mask_sentences = [tokens.copy() for _ in range(len(tokens))]
        for i in range(len(tokens)):
            mask_sentences[i][i] = self.tokenizer.mask_token
        return [[self.tokenizer.cls_token] + mask_sentence + [self.tokenizer.sep_token]
                for mask_sentence in mask_sentences]

    # @overrides
    def _tokens_log_prob(
        self, text: str
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, List[str]]:
        device = self.model.device

        tokens = self.tokenizer.tokenize(text)

        # Store the full encoded sentences it to easily retrieve the token score in mask scores sentences
        encoded_sentence = self.tokenizer.convert_tokens_to_ids(tokens)
        seq_len = len(tokens)
        mask_tok_sentences = self._generate_mask_sentences(tokens)

        # ids.shape = [seq_len, seq_len + 2]
        ids = torch.stack([torch.tensor(self.tokenizer.convert_tokens_to_ids(mask_tok_sentence),
                                        device=device,
                                        dtype=torch.long)
                           for mask_tok_sentence in mask_tok_sentences],
                          dim=0)

        # For now, I pass all the mask sentences in one single batch
        with torch.no_grad():
            outputs = self.model(ids)

        # pred_scores.shape = [seq_len, seq_len + 2, vocab_size]
        pred_scores = outputs[0]

        # retrieve only logits corresponding to mask tokens :
        mask_positions = range(1, 1 + seq_len)  # not take into account first and last special tokens
        mask_pred_logits = pred_scores[range(seq_len), mask_positions, :]  # shape (seq_len, vocab_size)
        tokens_scores = mask_pred_logits[range(seq_len), encoded_sentence]  # shape (seq_len, )

        log_probs = tokens_scores - mask_pred_logits.logsumexp(dim=1)  # shape (seq_len, )

        return log_probs, torch.tensor(encoded_sentence), tokens

    # @overrides
    @classmethod
    def _supported_model_names(cls) -> Iterable[str]:
        return BertForMaskedLM.pretrained_model_archive_map.keys()
