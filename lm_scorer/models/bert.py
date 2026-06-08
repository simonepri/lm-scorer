from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForMaskedLM, AutoTokenizer
from transformers.tokenization_utils import BatchEncoding

from .abc.transformers import TransformersLMScorer


class BERTLMScorer(TransformersLMScorer):
    """
    Use BERT to score a sentence following the idea describe in the paper
    Effective Sentence Scoring Method Using BERT for Speech Recognition. J. Shin, Y. Lee, Kyomin Jung

    The idea is to mask successively each token in the sentences and compute the log prob of true token.

    For instance, if the sentence to score is "I like tennis":
        1- Create the following masked sentences
        2- Compute the log-likelihood of each target word that has been masked using context from both sides
            - [CLS] [MASK]  like  tennis [SEP]  -> P_bert(tok[1] == I) ?
            - [CLS]   I    [MASK] tennis [SEP]  -> P_bert(tok[2] == like) ?
            - [CLS]   I     like  [MASK] [SEP]  -> P_bert(tok[3] == tennis) ?
    """

    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        # pylint: disable=attribute-defined-outside-init
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, pad_token="<|pad|>"
        )

        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        if "device" in options:
            self.model.to(options["device"])

    def _generate_batch_of_mask_sentences(self, text: List[str]) -> Iterator[Dict]:
        """
        Generate on the fly the mask sentences
        Once batch_size mask sentences has been generated, yield a batch as a dict containing :
        - mask_ids : List[torch.Tensor] -> token_ids of each masked sentence
        - mask_positions : List[int] -> position of the mask in each masked sentence
        - target_ids: List[int] -> token_id that has been masked in each masked sentence
        - target_tokens: List[str] -> token thas has been masked in each masked sentence
        - sentence_index : List[int] -> index w.r.t. text of the sentence that has been masked
        """

        batch: Dict[str, List] = {
            "mask_ids": [],
            "mask_positions": [],
            "target_ids": [],
            "target_tokens": [],
            "sentence_index": [],
        }

        for sent_index, sentence in enumerate(text):
            encoding: BatchEncoding = self.tokenizer.encode_plus(
                sentence, add_special_tokens=True, return_tensors="pt",
            )

            sent_ids = encoding["input_ids"].to(self.model.device)

            # Create masked sentences by successively masking
            # each token except CLS and SEP special tokens
            for i in range(1, sent_ids.shape[1] - 1):
                mask_ids = sent_ids.detach().clone().view(-1)
                mask_ids[i] = self.tokenizer.mask_token_id

                new_mask_sentences = {
                    "mask_ids": mask_ids,
                    "mask_positions": i,
                    "target_ids": sent_ids[0, i].item(),
                    "target_tokens": encoding.tokens(0)[i],
                    "sentence_index": sent_index,
                }

                # Add the mask sentence and its features to the current batch
                for key in new_mask_sentences:
                    batch[key].append(new_mask_sentences[key])

                # When batch size has reached batch_size, yield it and re-initialize it
                if len(batch["mask_ids"]) == self.batch_size:
                    yield batch
                    for key in batch:
                        batch[key] = []

        # Yield remaining mask sentences
        if len(batch["mask_ids"]) > 0:
            yield batch

    def _mask_tokens_log_prob_for_batch(self, batch: Dict[str, List]) -> List[float]:
        """
        Given a batch, compute and return the log prob of target token in each mask sentences
        """
        batch_size = len(batch["mask_ids"])
        pad_mask_ids = pad_sequence(
            batch["mask_ids"],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_mask_ids != self.tokenizer.pad_token_id

        with torch.no_grad():
            logits: torch.Tensor = self.model(
                pad_mask_ids, attention_mask=attention_mask
            )[0]

        # Retrieve the logits of mask tokens
        # mask_pred_logits.shape = [batch_size, vocac_size]
        mask_pred_logits = logits[range(batch_size), batch["mask_positions"], :]

        # target_score.shape = [batch_size,]
        target_scores = mask_pred_logits[range(batch_size), batch["target_ids"]]
        target_log_probs = target_scores - mask_pred_logits.logsumexp(dim=1)

        return target_log_probs.tolist()

    @staticmethod
    def _gather_result_by_sentence(
        result: Dict[str, List]
    ) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:

        outputs: List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]] = []
        mask_to_sent_idx = result["sentence_index"]
        for sent_idx in set(mask_to_sent_idx):
            begin_idx = mask_to_sent_idx.index(sent_idx)
            end_idx = len(mask_to_sent_idx) - mask_to_sent_idx[::-1].index(sent_idx)

            sent_log_probs = torch.tensor(  # pylint: disable=not-callable
                result["target_log_probs"][begin_idx:end_idx]
            )
            sent_ids = torch.tensor(  # pylint: disable=not-callable
                result["target_ids"][begin_idx:end_idx]
            )

            sent_log_probs = cast(torch.DoubleTensor, sent_log_probs)
            sent_ids = cast(torch.LongTensor, sent_ids)
            sent_tokens: List[str] = result["target_tokens"][begin_idx:end_idx]

            outputs.append((sent_log_probs, sent_ids, sent_tokens))

        return outputs

    def _tokens_log_prob(
        self, text: List[str]
    ) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        result: Dict[str, List] = {
            "target_log_probs": [],
            "target_ids": [],
            "target_tokens": [],
            "sentence_index": [],
        }

        # Compute mask token score by batch of batch_size
        for batch in self._generate_batch_of_mask_sentences(text):
            batch["target_log_probs"] = self._mask_tokens_log_prob_for_batch(batch)
            for key in result:
                result[key].extend(batch[key])

        return self._gather_result_by_sentence(result)

    def _tokens_log_prob_for_batch(
        self, text: List[str]
    ) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        # Because masked sentences are generated,
        # the number of sentences that will be input in the LM are not known in advance
        # As a result, this scorer do not use the BatchedLMScorer structure
        ...

    # @overrides
    @classmethod
    def _supported_model_names(cls) -> Iterable[str]:
        return BertForMaskedLM.pretrained_model_archive_map.keys()
