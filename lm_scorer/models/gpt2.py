from typing import Any, Dict, Iterable, List, Tuple, cast

import torch
from transformers import AutoTokenizer, BatchEncoding, GPT2LMHeadModel

from .abc.transformers import TransformersLMScorer


class GPT2LMScorer(TransformersLMScorer):
    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        # Whether to prepend/append the bos/eos tokens when scoring. Appending
        # the eos token (the default) also scores the probability that the
        # sentence ends; pass eos=False to omit it from the score (see #12).
        self.add_bos = options.get("bos", True)
        self.add_eos = options.get("eos", True)

        # Forward any remaining options to the Hugging Face loaders so callers
        # can pass e.g. local_files_only, revision, cache_dir, token or
        # torch_dtype (see #8).
        hf_kwargs = {
            key: value
            for key, value in options.items()
            if key not in ("batch_size", "device", "bos", "eos")
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, **{"use_fast": True, **hf_kwargs}
        )
        # Add the pad token to GPT2 dictionary.
        # len(tokenizer) = vocab_size + 1
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
        self.tokenizer.pad_token = "<|pad|>"

        self.model = GPT2LMHeadModel.from_pretrained(model_name, **hf_kwargs)
        # We need to resize the embedding layer because we added the pad token.
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        if "device" in options:
            self.model.to(options["device"])

    def _add_special_tokens(self, text: str) -> str:
        bos = self.tokenizer.bos_token if self.add_bos else ""
        eos = self.tokenizer.eos_token if self.add_eos else ""
        return bos + text + eos

    # @overrides
    def _tokens_log_prob_for_batch(
        self, text: List[str]
    ) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        outputs: List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]] = []
        if len(text) == 0:
            return outputs

        # TODO: Handle overflowing elements for long sentences
        text = list(map(self._add_special_tokens, text))
        encoding: BatchEncoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            ids = encoding["input_ids"].to(self.model.device)
            attention_mask = encoding["attention_mask"].to(self.model.device)
            nopad_mask = ids != self.tokenizer.pad_token_id
            logits: torch.Tensor = self.model(ids, attention_mask=attention_mask)[0]

        for sent_index in range(len(text)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(encoding.tokens(sent_index))
                if sent_nopad_mask[i] and i != 0
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")
            # ids_scores.shape = [seq_len + 1]
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            # log_prob.shape = [seq_len + 1]
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)

            sent_log_probs = cast(torch.DoubleTensor, sent_log_probs)
            sent_ids = cast(torch.LongTensor, sent_ids)

            output = (sent_log_probs, sent_ids, sent_tokens)
            outputs.append(output)

        return outputs

    # @overrides
    @classmethod
    def _supported_model_names(cls) -> Iterable[str]:
        return ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2"]
