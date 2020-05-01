from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn.utils.rnn import pad_sequence

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

        with torch.no_grad():
            outputs = self.model(ids)

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

    # @overrides_tokens_log_prob_single_batch
    @classmethod
    def _supported_model_names(cls) -> Iterable[str]:
        return GPT2LMHeadModel.pretrained_model_archive_map.keys()
