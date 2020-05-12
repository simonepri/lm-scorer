from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import


import torch
<<<<<<< HEAD
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.utils.rnn import pad_sequence
=======
from transformers import AutoTokenizer, AutoConfig, GPT2LMHeadModel
from transformers.tokenization_utils import BatchEncoding
>>>>>>> gpt2_batching_new_API

from .abc.transformers import TransformersLMScorer


class GPT2LMScorer(TransformersLMScorer):
    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        # pylint: disable=attribute-defined-outside-init
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, add_special_tokens=False
        )
        # Add the pad token to GPT2 dictionary.
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
        self.tokenizer.pad_token = "<|pad|>"

        # Index to easily access all logits value except the one corresponding to pad_token_id
        self.nopad_vocab_idx = [
            *range(self.tokenizer.pad_token_id),
            *range(self.tokenizer.pad_token_id + 1, len(self.tokenizer)),
        ]

        config = AutoConfig.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # We need to resize the embedding layer because we added the pad token.
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        if "device" in options:
            self.model.to(options["device"])

<<<<<<< HEAD
        self.batch_size = options["batch_size"] if "batch_size" in options else 1

    def _tokens_log_prob_single_batch(
        self, text: List[str]
    ) -> List[Tuple[torch.FloatTensor, torch.LongTensor, List[str]]]:
        device = self.model.device

        input_text = [
            self.tokenizer.bos_token + sentence + self.tokenizer.eos_token
            for sentence in text
        ]

        tokens_list = [self.tokenizer.tokenize(sentence) for sentence in input_text]

        tokens_ids_list = [
            torch.tensor(  # pylint: disable=not-callable
                self.tokenizer.convert_tokens_to_ids(tokens),
                device=device,
                dtype=torch.long,
            )
            for tokens in tokens_list
        ]

        ids = pad_sequence(
            tokens_ids_list, batch_first=True, padding_value=self.tokenizer.eos_token_id
        )
=======
    def _add_special_tokens(self, text: str) -> str:
        return self.tokenizer.bos_token + text + self.tokenizer.eos_token

    # @overrides
    def _tokens_log_prob_for_batch(
        self, text: List[str]
    ) -> List[Tuple[torch.FloatTensor, torch.LongTensor, List[str]]]:
        outputs: List[Tuple[torch.FloatTensor, torch.LongTensor, List[str]]] = []
        if len(text) == 0:
            return outputs
>>>>>>> gpt2_batching_new_API

        # TODO: Handle overflowing elements for long sentences
        text = list(map(self._add_special_tokens, text))
        encoding: BatchEncoding = self.tokenizer.batch_encode_plus(
            text, return_tensors="pt",
        )
        with torch.no_grad():
            ids = encoding["input_ids"].to(self.model.device)
            nopad_mask = ids != self.tokenizer.pad_token_id
            logits: torch.Tensor = self.model(ids)[0]

<<<<<<< HEAD
        # pred_scores.shape = [nb_sentences, max_seq_len, vocab_size]
        pred_scores = outputs[0]

        # Align input and target
        ids = ids[:, 1:]
        pred_scores = pred_scores[:, :-1, :]

        # Retrieve the token scores corresponding to the target id
        # ids_scores.shape = [nb_sentences, max_seq_len]
        ids_scores = pred_scores.gather(2, ids.unsqueeze(2)).squeeze(2)

        # log_prob.shape = [nb_sentences, max_seq_len]
        log_probs = ids_scores - pred_scores.logsumexp(2)

        outputs = []
        for i in range(len(text)):
            output = (
                log_probs[i, : len(tokens_list[i]) - 1],
                ids[i, : len(tokens_list[i]) - 1],
                tokens_list[i][1:],
            )
            outputs.append(output)

        return outputs  # type: ignore
    
    # @overrides
    def _tokens_log_prob(
        self, sentences: List[str]
    ) -> List[Tuple[torch.FloatTensor, torch.LongTensor, List[str]]]:

        output = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            output += self._tokens_log_prob_single_batch(batch)

        if len(sentences) % self.batch_size != 0:
            remaining_batch = sentences[-(len(sentences) % self.batch_size) :]
            output += self._tokens_log_prob_single_batch(remaining_batch)

        return output
=======
        for sent_index in range(len(text)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = encoding.tokens(sent_index)[1:]
            # sent_ids.shape = [len(text[sent_index]) + 1]
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, self.nopad_vocab_idx]
            # ids_scores.shape = [seq_len + 1]
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            x = -sent_logits.logsumexp(1)
            # log_prob.shape = [seq_len + 1]
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)

            sent_log_probs = cast(torch.FloatTensor, sent_log_probs)
            sent_ids = cast(torch.LongTensor, sent_ids)

            output = (sent_log_probs, sent_ids, sent_tokens)
            outputs.append(output)

        return outputs
>>>>>>> gpt2_batching_new_API

    # @overrides
    @classmethod
    def _supported_model_names(cls) -> Iterable[str]:
        return GPT2LMHeadModel.pretrained_model_archive_map.keys()
