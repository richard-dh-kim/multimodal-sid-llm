"""Beam-search retriever for SID-LLM: text or query-embedding -> top-K item ids."""
from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn
from transformers import LogitsProcessorList, T5ForConditionalGeneration, T5TokenizerFast

from sid_llm.inference.trie import SIDTrie
from sid_llm.inference.logits_processor import TrieConstrainedSIDProcessor


class BeamSearchRetriever:
    """T5 + tokenizer + Trie + sid_to_item wrapped for generative retrieval.

    Pass `query_projection` and `soft_prompt_offsets` together to enable
    `retrieve_from_query_embedding`; otherwise only `retrieve_from_text` works.
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
        query_projection: nn.Linear | None = None,
        soft_prompt_offsets: torch.Tensor | nn.Parameter | None = None,
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

        self.query_projection: nn.Linear | None = None
        self.soft_prompt_offsets: torch.Tensor | None = None
        if query_projection is not None and soft_prompt_offsets is not None:
            self.query_projection = query_projection.to(self.device).eval()
            offsets = soft_prompt_offsets
            if isinstance(offsets, nn.Parameter):
                offsets = offsets.data
            self.soft_prompt_offsets = offsets.detach().to(self.device)
        elif (query_projection is None) != (soft_prompt_offsets is None):
            raise ValueError(
                "query_projection and soft_prompt_offsets must be provided together "
                "(or both omitted) to enable search-mode retrieval."
            )

        search_id = self.tokenizer.convert_tokens_to_ids("<search>")
        self._search_token_id: int | None = (
            search_id if search_id != self.tokenizer.unk_token_id else None
        )

    def _decode_sid_tuples_from_sequences(
        self, sequences: torch.LongTensor
    ) -> list[tuple[int, int, int, int]]:
        """[B*K, T] decoder sequences -> (cb_0, cb_1, cb_2, cb_3) per row.

        Rows shorter than 4 SID tokens get padded with -1 so they miss the
        sid_to_item lookup.
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

    def _build_search_inputs_embeds(
        self, query_embed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, query_dim] -> (inputs_embeds[B, N+1, d_model], attention_mask[B, N+1])."""
        assert self.query_projection is not None
        assert self.soft_prompt_offsets is not None
        assert self._search_token_id is not None

        b = query_embed.size(0)
        projected = self.query_projection(query_embed)
        soft = projected.unsqueeze(1) + self.soft_prompt_offsets.unsqueeze(0)

        search_ids = torch.full(
            (b, 1), fill_value=self._search_token_id, dtype=torch.long, device=self.device
        )
        search_emb = self.model.shared(search_ids)
        # Cast soft to embedding dtype so concat works under bf16/fp16.
        soft = soft.to(search_emb.dtype)
        inputs_embeds = torch.cat([soft, search_emb], dim=1)
        attention_mask = torch.ones(
            (b, inputs_embeds.size(1)), dtype=torch.long, device=self.device
        )
        return inputs_embeds, attention_mask

    @torch.no_grad()
    def retrieve_from_text(
        self,
        query_text: str,
        k: int = 10,
        num_beams: int | None = None,
        constrained: bool = True,
    ) -> tuple[list[int], list[tuple[int, int, int, int]]]:
        """Encode query, beam-search top-K SID sequences, map to item_ids.

        Returns (item_ids, sid_tuples), both length k, ordered by beam score.
        Lookup misses become item_id=-1.
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
            max_new_tokens=5,
            num_beams=num_beams,
            num_return_sequences=k,
            do_sample=False,
            logits_processor=logits_processors if constrained else LogitsProcessorList(),
            return_dict_in_generate=True,
            use_cache=True,
        )

        sid_tuples = self._decode_sid_tuples_from_sequences(out.sequences)
        item_ids = [self.sid_to_item.get(t, -1) for t in sid_tuples]
        return item_ids, sid_tuples

    @torch.no_grad()
    def retrieve_from_query_embedding(
        self,
        query_embed: torch.Tensor,
        k: int = 10,
        num_beams: int | None = None,
        constrained: bool = True,
    ) -> (
        tuple[list[int], list[tuple[int, int, int, int]]]
        | tuple[list[list[int]], list[list[tuple[int, int, int, int]]]]
    ):
        """Search-mode beam search from a CLIP query embedding.

        query_embed of shape [D] returns (item_ids, sid_tuples) of length k.
        Shape [B, D] returns lists-of-lists, one per query.
        """
        if self.query_projection is None or self.soft_prompt_offsets is None:
            raise RuntimeError(
                "retrieve_from_query_embedding requires query_projection + "
                "soft_prompt_offsets; this retriever was loaded without search-mode "
                "soft prompts (sequence-mode-only checkpoint)."
            )
        if self._search_token_id is None:
            raise RuntimeError(
                "<search> token is not in the tokenizer vocabulary; the loaded "
                "checkpoint cannot drive search-mode retrieval."
            )

        single = (query_embed.dim() == 1)
        if single:
            query_embed = query_embed.unsqueeze(0)
        if query_embed.dim() != 2:
            raise ValueError(
                f"query_embed must be 1D or 2D; got shape {tuple(query_embed.shape)}"
            )

        target_dtype = self.query_projection.weight.dtype
        query_embed = query_embed.to(device=self.device, dtype=target_dtype)

        num_beams = num_beams or max(k, 4)

        inputs_embeds, attention_mask = self._build_search_inputs_embeds(query_embed)

        logits_processors = LogitsProcessorList()
        if constrained:
            logits_processors.append(
                TrieConstrainedSIDProcessor(
                    self.trie, self.sid_token_ids, decoder_start_offset=1
                )
            )

        out = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=5,
            num_beams=num_beams,
            num_return_sequences=k,
            do_sample=False,
            logits_processor=logits_processors if constrained else LogitsProcessorList(),
            return_dict_in_generate=True,
            use_cache=True,
        )

        all_sids = self._decode_sid_tuples_from_sequences(out.sequences)
        all_items = [self.sid_to_item.get(t, -1) for t in all_sids]

        b = query_embed.size(0)
        if single:
            return all_items[:k], all_sids[:k]

        per_query_items: list[list[int]] = []
        per_query_sids: list[list[tuple[int, int, int, int]]] = []
        for i in range(b):
            per_query_items.append(all_items[i * k : (i + 1) * k])
            per_query_sids.append(all_sids[i * k : (i + 1) * k])
        return per_query_items, per_query_sids


def load_retriever(
    ckpt_dir: Path,
    sid_to_item_path: Path,
    sid_trie_path: Path,
    soft_prompt_path: Path | None = None,
    device: str | None = None,
) -> BeamSearchRetriever:
    """Load a SID-LLM checkpoint plus catalog lookup artifacts.

    If `soft_prompt.pt` is at `soft_prompt_path` (default
    `<ckpt_dir>/soft_prompt.pt`), the retriever supports search mode. Otherwise
    `retrieve_from_query_embedding` will raise.
    """
    import pickle

    ckpt_dir = Path(ckpt_dir)
    tokenizer = T5TokenizerFast.from_pretrained(str(ckpt_dir))
    model = T5ForConditionalGeneration.from_pretrained(str(ckpt_dir))

    with open(sid_to_item_path, "rb") as f:
        sid_to_item = pickle.load(f)
    with open(sid_trie_path, "rb") as f:
        trie: SIDTrie = pickle.load(f)

    sid_token_ids = tokenizer.convert_tokens_to_ids(
        [f"<sid_{i}>" for i in range(1024)]
    )
    sid_eos_id = tokenizer.convert_tokens_to_ids("<sid_eos>")

    qp: nn.Linear | None = None
    offsets: torch.Tensor | None = None
    sp_path = Path(soft_prompt_path) if soft_prompt_path is not None else (ckpt_dir / "soft_prompt.pt")
    if sp_path.exists():
        sd = torch.load(str(sp_path), map_location="cpu", weights_only=False)
        qp_sd = sd.get("query_projection.state_dict")
        offsets_t = sd.get("soft_prompt_offsets")
        if qp_sd is not None and offsets_t is not None:
            in_features = qp_sd["weight"].shape[1]
            out_features = qp_sd["weight"].shape[0]
            qp = nn.Linear(in_features, out_features, bias="bias" in qp_sd)
            qp.load_state_dict(qp_sd)
            offsets = offsets_t.detach().clone()
        else:
            warnings.warn(
                f"{sp_path} found but missing required keys "
                "('query_projection.state_dict' and/or 'soft_prompt_offsets'); "
                "loading retriever in sequence-mode only.",
                stacklevel=2,
            )
    elif soft_prompt_path is not None:
        warnings.warn(
            f"soft_prompt path {sp_path} not found; loading retriever in "
            "sequence-mode only.",
            stacklevel=2,
        )

    return BeamSearchRetriever(
        model=model, tokenizer=tokenizer, trie=trie, sid_to_item=sid_to_item,
        sid_token_ids=sid_token_ids, sid_eos_id=sid_eos_id, device=device,
        query_projection=qp, soft_prompt_offsets=offsets,
    )
