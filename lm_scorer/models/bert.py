from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import torch
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
        self.batch_size = options["batch_size"] if "batch_size" in options else 1

    def _generate_mask_sentences(self, tokens: List[str]) -> List[List[str]]:
        mask_sentences = [tokens.copy() for _ in range(len(tokens))]
        for i in range(len(tokens)):
            mask_sentences[i][i] = self.tokenizer.mask_token
        return [
            [self.tokenizer.cls_token] + mask_sentence + [self.tokenizer.sep_token]
            for mask_sentence in mask_sentences
        ]

    # @overrides
    def _tokens_log_prob(
        self, text: str
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, List[str]]:
        if text == "":
            return (torch.zeros((1,)), torch.zeros((1,)), [""])  # type: ignore # pylint: disable=not-callable
        device = self.model.device
        tokens = self.tokenizer.tokenize(text)

        # Store the full encoded sentences it to easily retrieve the token score in mask scores sentences
        encoded_sentence = self.tokenizer.convert_tokens_to_ids(tokens)
        seq_len = len(tokens)
        mask_tok_sentences = self._generate_mask_sentences(tokens)

        # ids.shape = [seq_len, seq_len + 2]
        ids = torch.stack(
            [
                torch.tensor(  # pylint: disable=not-callable
                    self.tokenizer.convert_tokens_to_ids(mask_tok_sentence),
                    device=device,
                    dtype=torch.long,
                )
                for mask_tok_sentence in mask_tok_sentences
            ],
            dim=0,
        )

        # Compute all prediction logits by batch
        i = 0
        all_pred_scores = []
        with torch.no_grad():
            while i + self.batch_size < seq_len:
                all_pred_scores.append(self.model(ids[i : i + self.batch_size])[0])
                i += self.batch_size
            if i < seq_len:
                all_pred_scores.append(self.model(ids[i:])[0])

        # pred_scores.shape = [seq_len, seq_len + 2, vocab_size]
        pred_scores = torch.cat(all_pred_scores, dim=0)

        # retrieve only logits corresponding to mask tokens and do not take into account CLS and SEP special tokens.
        # mask_pred_logits.shape = [seq_len, vocab_size]
        mask_pred_logits = pred_scores[range(seq_len), range(1, 1 + seq_len), :]

        # tokens_scores.shape = [seq_len, ]
        tokens_scores = mask_pred_logits[range(seq_len), encoded_sentence]

        log_probs = tokens_scores - mask_pred_logits.logsumexp(dim=1)

        return log_probs, torch.tensor(encoded_sentence), tokens  # type: ignore # pylint: disable=not-callable

    # @overrides
    @classmethod
    def _supported_model_names(cls) -> Iterable[str]:
        return BertForMaskedLM.pretrained_model_archive_map.keys()
