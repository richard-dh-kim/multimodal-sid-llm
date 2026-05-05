"""High-level wrapper around HF model.generate() for SID-LLM retrieval."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import LogitsProcessorList, T5ForConditionalGeneration, T5TokenizerFast

from sid_llm.inference.trie import SIDTrie
from sid_llm.inference.logits_processor import TrieConstrainedSIDProcessor


class BeamSearchRetriever:
    """Wraps a T5 model + tokenizer + Trie + sid_to_item for generative retrieval.

    The model is expected to have been CPT'd (M3.6) and fine-tuned (M3.7) so that
    generated tokens are real SID tokens. With the M3.5 init checkpoint, generation
    will be near-random - useful only for testing the mechanics.
    """

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        trie: SIDTrie,
        sid_to_item: dict[tuple[int, int, int, int], int],
        sid_token_ids: list[int],
        sid_eos_id: int,
        device: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.trie = trie
        self.sid_to_item = sid_to_item
        self.sid_token_ids = sid_token_ids
        self.sid_eos_id = sid_eos_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.token_id_to_codebook_index: dict[int, int] = {
            tid: i for i, tid in enumerate(sid_token_ids)
        }

    def _decode_sid_tuples_from_sequences(
        self, sequences: torch.LongTensor
    ) -> list[tuple[int, int, int, int]]:
        """`sequences` shape [B*K, T]. Strip decoder_start_token + eos and translate
        to (cb_idx_0, cb_idx_1, cb_idx_2, cb_idx_3) tuples. Sequences shorter than
        4 SID tokens get padded with -1 (which won't match any sid_to_item key).
        """
        out: list[tuple[int, int, int, int]] = []
        for row in sequences.tolist():
            cb: list[int] = []
            for tid in row:
                if tid in self.token_id_to_codebook_index:
                    cb.append(self.token_id_to_codebook_index[tid])
                    if len(cb) == 4:
                        break
            while len(cb) < 4:
                cb.append(-1)
            out.append(tuple(cb[:4]))
        return out

    @torch.no_grad()
    def retrieve_from_text(
        self,
        query_text: str,
        k: int = 10,
        num_beams: int | None = None,
        constrained: bool = True,
    ) -> tuple[list[int], list[tuple[int, int, int, int]]]:
        """Encode query, beam-search top-K SID sequences, map to item_ids.

        Returns (item_ids, sid_tuples) - both length up to k. Order is by beam score (best first).
        Items that fail dict lookup get item_id -1 (silent miss).
        """
        num_beams = num_beams or max(k, 4)
        enc = self.tokenizer(query_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        logits_processors = LogitsProcessorList()
        if constrained:
            logits_processors.append(
                TrieConstrainedSIDProcessor(
                    self.trie, self.sid_token_ids, decoder_start_offset=1
                )
            )

        out = self.model.generate(
            **enc,
            max_new_tokens=5,  # 4 SID tokens + maybe eos
            num_beams=num_beams,
            num_return_sequences=k,
            do_sample=False,
            logits_processor=logits_processors if constrained else LogitsProcessorList(),
            return_dict_in_generate=True,
            use_cache=True,  # KV cache enabled
        )

        sid_tuples = self._decode_sid_tuples_from_sequences(out.sequences)
        item_ids = [self.sid_to_item.get(t, -1) for t in sid_tuples]
        return item_ids, sid_tuples


def load_retriever(
    ckpt_dir: Path,
    sid_to_item_path: Path,
    sid_trie_path: Path,
    soft_prompt_path: Path | None = None,
    device: str | None = None,
) -> BeamSearchRetriever:
    """Load a SID-LLM checkpoint + the M3.4 lookup artifacts."""
    import pickle

    tokenizer = T5TokenizerFast.from_pretrained(str(ckpt_dir))
    model = T5ForConditionalGeneration.from_pretrained(str(ckpt_dir))

    with open(sid_to_item_path, "rb") as f:
        sid_to_item = pickle.load(f)
    with open(sid_trie_path, "rb") as f:
        trie: SIDTrie = pickle.load(f)

    # Resolve sid_token_ids from the tokenizer (preferred) or soft_prompt extras.
    sid_token_ids = tokenizer.convert_tokens_to_ids(
        [f"<sid_{i}>" for i in range(1024)]
    )
    sid_eos_id = tokenizer.convert_tokens_to_ids("<sid_eos>")

    return BeamSearchRetriever(
        model=model, tokenizer=tokenizer, trie=trie, sid_to_item=sid_to_item,
        sid_token_ids=sid_token_ids, sid_eos_id=sid_eos_id, device=device,
    )
