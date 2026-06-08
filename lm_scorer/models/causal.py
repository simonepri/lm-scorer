from typing import Any, Dict, Iterable, List, Tuple, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from .abc.transformers import TransformersLMScorer


class CausalLMScorer(TransformersLMScorer):
    """Score sentences with any causal (autoregressive) language model.

    Works with any model loadable through ``AutoModelForCausalLM`` (GPT-2,
    GPT-Neo, OPT, BLOOM, Llama, multilingual GPT-2s, ...). Unlike the GPT-2
    specific scorer it does not add a dedicated pad token nor resize the
    embeddings: it pads with the eos token and relies on the attention mask to
    tell real tokens from padding, which avoids vocabulary/size mismatches on
    models whose embeddings are not resized in lockstep.
    """

    # @overrides
    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        super()._build(model_name, options)

        # Whether to prepend/append the bos/eos tokens when scoring (see #12).
        self.add_bos = options.get("bos", True)
        self.add_eos = options.get("eos", True)

        # Forward any remaining options to the Hugging Face loaders (see #8).
        hf_kwargs = {
            key: value
            for key, value in options.items()
            if key not in ("batch_size", "device", "bos", "eos")
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, **{"use_fast": True, **hf_kwargs}
        )
        # Causal LMs are scored left-to-right, so pad on the right and reuse the
        # eos token as pad when the tokenizer has none (no embedding resize).
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **hf_kwargs)
        self.model.eval()
        if "device" in options:
            self.model.to(options["device"])

    def _add_special_tokens(self, text: str) -> str:
        bos = (
            self.tokenizer.bos_token
            if self.add_bos and self.tokenizer.bos_token
            else ""
        )
        eos = (
            self.tokenizer.eos_token
            if self.add_eos and self.tokenizer.eos_token
            else ""
        )
        return bos + text + eos

    # @overrides
    def _tokens_log_prob_for_batch(
        self, text: List[str]
    ) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        outputs: List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]] = []
        if len(text) == 0:
            return outputs

        text = list(map(self._add_special_tokens, text))
        encoding: BatchEncoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        with torch.no_grad():
            ids = encoding["input_ids"].to(self.model.device)
            attention_mask = encoding["attention_mask"].to(self.model.device)
            # The attention mask (not the pad token id) identifies real tokens,
            # so a real trailing eos is not mistaken for padding.
            nopad_mask = attention_mask.bool()
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
        # Not restricted to a fixed list: any AutoModelForCausalLM model works.
        return []
