from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from torch.nn.utils.rnn import pad_sequence

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

        self.batch_size = options["batch_size"] if "batch_size" in options else 1

    def add_special_tokens_and_encode(self, text):
        return self.tokenizer.encode(self.tokenizer.bos_token + text + self.tokenizer.eos_token)

    def pad(self, sequences: List[torch.Tensor]):
        max_seq_len = max([s.size(0) for s in sequences])
        out_tensor = sequences[0].data.new(len(sequences), max_seq_len).fill_(self.tokenizer.eos_token_id)
        mask = torch.zeros((len(sequences), max_seq_len), device=sequences[0].device)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1

        return out_tensor, mask

    def _tokens_log_prob_single_batch(
            self, sentences: List[str]
    ) -> List[Tuple[torch.FloatTensor, torch.LongTensor, List[str]]]:

        device = self.model.device

        tokens = [self.add_special_tokens_and_encode(sentence) for sentence in sentences]
        ids, mask = self.pad(list(map(lambda x: torch.tensor(x, device=device, dtype=torch.long), tokens)))

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

        # Zero the values of the padding inputs
        ids_scores *= mask[:, 1:]

        # log_prob.shape = [nb_sentences, max_seq_len]
        log_probs = ids_scores - pred_scores.logsumexp(2)

        return [(log_probs[i, :len(tokens[i])], ids[i, :len(tokens[i])], tokens[i]) for i in range(len(sentences))]

    def _tokens_log_prob(self, sentences):
        output = []
        for i in range(len(sentences) // self.batch_size):
            output += self._tokens_log_prob_single_batch(sentences[i * self.batch_size: (i+1) * self.batch_size])
        if len(sentences) % self.batch_size != 0:
            output += self._tokens_log_prob_single_batch(sentences[- (len(sentences) % self.batch_size):])
        return output

    # @overrides_tokens_log_prob_single_batch
    @classmethod
    def _supported_model_names(cls) -> Iterable[str]:
        return GPT2LMHeadModel.pretrained_model_archive_map.keys()
