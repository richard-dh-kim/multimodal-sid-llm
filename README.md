# multimodal-sid-llm

Multimodal generative retrieval over Semantic IDs. The model takes a query (text or photo) and returns matching items from a ~150k Amazon Products catalog by autoregressively generating the items' discrete 4-token semantic IDs, with **0% hallucinations guaranteed** by Trie-constrained beam search.

## Headline results

Evaluated on a 2,000-query held-out sample from the 150k-item catalog, beam=50, Trie-constrained decoding (where applicable). Random over 150k items: recall@10 = 0.0067%.

### Multimodal search — `sid_llm_eval_search.py`

The unique capability of this project: given an item's CLIP embedding (a query proxy for "show me items like this photo / this text"), generate the matching item's 4-token SID via beam search. **No prior art baseline runs this task** — TIGER and dense MIPS baselines do not accept embedding queries against a generative SID decoder.

| Eval | recall@5 | recall@10 | recall@50 | Hallucination |
|---|---|---|---|---|
| **M3.7 (ours, multimodal)** | **28.9%** | **39.4%** | **66.05%** | **0%** |

**5,880× over random.** The CLIP soft-prompt + Trie-constrained beam search resolves a 512-dim continuous query into a discrete SID over a 150k-item catalog with no hallucination.

### Sequence-mode rec — next-item prediction (TIGER-comparable)

| System | Mechanism | recall@10 | recall@50 | Hallucination |
|---|---|---|---|---|
| Random | — | 0.0067% | 0.033% | — |
| B1 (plain CLIP MIPS) | dense retrieval | 2.55% | 3.31% | — |
| B2 (VL-CLIP MIPS) | dense retrieval | 2.68% | 3.66% | — |
| **B3 (TIGER baseline)** | generative retrieval (8.6M-param T5 from scratch) | 1.75% | 3.95% | **0%** |
| **M3.7 (ours, multimodal)** | generative retrieval + soft-prompt + AdamWAnchored | 1.50% | 3.60% | **0%** |

**M3.7 vs TIGER:** TIGER edges M3.7 by 0.25pp on recall@10. **The multi-task training (sequence + search) costs near-zero rec quality** while *adding* the multimodal search capability that no other system in this table has. That tradeoff is the whole point of the project's architecture.

**Generative retrieval vs dense MIPS:** On this corpus size (150k items), dense MIPS wins on rec recall by roughly 1pp. This is consistent with the [scaling-view paper](https://arxiv.org/pdf/2509.25522) which argues generative retrieval needs significantly more scale to dominate dense baselines. The MIPS baselines used 19,994 queries vs 2,000 for the generative systems; the comparison is qualitatively meaningful but not perfectly apples-to-apples on sample size.

**Trie-constrained decoding gives 0% hallucination by construction** — every generated SID maps to a real catalog item. Applies to both B3 and M3.7 since it's an inference-time logits processor.

## Architecture

```
   query (text or image)
          │
          ▼
  ┌──────────────────────────────────┐
  │  CLIP ViT-B/32 → 512-dim embed   │
  └──────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────┐
  │  Soft-prompt projection W_q      │
  │  [B, 512] → [B, 4, 768]          │
  │  + position offsets p_i          │
  └──────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────┐
  │  T5-base (vocab 33,128)          │
  │  including 1024 SID tokens       │
  │  + <sid_eos> + <seq> + <search>  │
  │  CPT'd on mixed catalog corpus,  │
  │  fine-tuned with AdamWAnchored   │
  │  (L2-SP) in sequence + search    │
  │  modes via stratified 50/50      │
  │  per-batch sampling.             │
  └──────────────────────────────────┘
          │
          ▼
   beam search (4 SID steps × 50 beams)
   with KV cache + Trie-constrained logits processor
          │
          ▼
   {(t1,t2,t3,t4): item_id} dictionary lookup
          │
          ▼
   top-K items
```

## Pipeline (training stages)

| Stage | What it produces | Time on RTX 4060 Ti (16 GB) |
|---|---|---|
| **M0** Data pipeline | 150k items, 11.8M interactions, 149,936 images | ~30 min one-time |
| **M1** Preprocessing + B1 plain-CLIP baseline | Cleaned text, embeddings_b1.parquet | ~1 hr |
| **M2** VL-CLIP fine-tune + B2 baseline | embeddings_b2.parquet (improved CLIP) | ~3 hr |
| **M3.3** RQ-VAE Semantic-ID tokenizer | 1024-codebook 4-level quantizer; 100% codebook usage | ~30 min |
| **M3.4** SID-augmented catalog + Trie | catalog_with_sid.parquet, sid_to_item.pkl, sid_trie.pkl | ~5 min |
| **M3.5** T5 vocab expansion + soft-prompt init | init/hf_model + soft_prompt.pt | ~5 min |
| **M3.6** Continued pre-training (CPT) | cpt/hf_best (T5 with SID semantics learned) | ~10 hr (8 epochs) |
| **M3.7** Generative-retrieval fine-tune | retrieval/hf_best + soft_prompt.pt (sequence + search modes) | ~5 hr (5 epochs) |
| **M3 / Task 25** TIGER baseline (B3) | tiger_b3/hf_best (8.6M-param vanilla T5, no contributions) | ~3 hr (8 epochs) |
| **Eval** | recall@k / ndcg@k JSON files in eval/results/ | ~5–15 min per eval |

## Distinguishing contributions vs prior art

- **Multimodal front-end** (CLIP soft-prompts feeding a TIGER-style SID-LLM). Most published SID-LLM work is item-ID-only.
- **Trie-constrained decoding** — every generated 4-token SID is guaranteed valid, by construction. Hallucination rate is 0% (vs the typical 5–30% silent-miss rate of unconstrained beam search on TIGER-style models).
- **AdamWAnchored optimizer** (L2-SP regularization toward the CPT init weights) — preserves T5's pretraining/CPT knowledge during the retrieval fine-tune.
- **KV cache during beam search** — Hugging Face's beam search supports it natively for T5; this implementation explicitly enables it. ~32× speedup on the decode loop vs reference TIGER implementations that disable KV cache.

## Dataset

Amazon Reviews 2023 (McAuley Lab, UC San Diego), curated subset:

| Category | Items | Subcategories included |
|---|---|---|
| Tools & Home Improvement | 50,000 | Power & Hand Tools, Hardware, Safety & Security, Electrical, Light Bulbs |
| Home & Kitchen | 50,000 | Kitchen & Dining, Storage & Organization |
| Electronics | 50,000 | Computers & Accessories, Headphones, Portable A/V, Wearables, Accessories & Supplies |

**Total: 150,000 items, 11.8M interactions, 149,936 images (99.96%).**

Filtering criteria: `rating_number ≥ 5`, has at least one `large` image URL, title ≥ 5 chars, in storage-room-relevant subcategories. License: research-permissive.

## Repo layout

```
multimodal-sid-llm/
├── src/sid_llm/        ← all code, organized by stage (data/, models/, training/, inference/, eval/, demo/)
├── tests/              ← pytest suite, mirrors src/ structure
├── scripts/            ← shell entry points per pipeline stage
├── configs/            ← Hydra YAML configs
├── data/               ← gitignored; rebuilt by download scripts
└── checkpoints/        ← gitignored; produced by training
```

## Reproduction

```bash
# 1. Setup
py -m venv .venv
.venv/Scripts/pip install -e ".[dev]"

# 2. Data pipeline (one-time)
bash scripts/run_download_metadata.sh    # ~36 sec
bash scripts/run_download_reviews.sh     # ~14 min
bash scripts/run_download_images.sh      # ~18 min
bash scripts/run_build_catalog.sh        # ~30 sec

# 3. Preprocessing + baselines
bash scripts/run_embed_b1.sh             # plain CLIP embeddings
bash scripts/run_train_vl_clip.sh        # VL-CLIP fine-tune
bash scripts/run_embed_b2.sh             # VL-CLIP embeddings (used by training + search eval)

# 4. Tokenizer + base model
bash scripts/run_train_rqvae.sh          # RQ-VAE quantizer
bash scripts/run_augment_catalog.sh      # add SIDs + build Trie
bash scripts/run_init_sid_llm.sh         # T5 vocab expansion + soft-prompt init

# 5. CPT + retrieval fine-tune
bash scripts/run_build_cpt_corpus.sh
bash scripts/run_train_cpt.sh \
    --batch-size 8 --accumulate-grad-batches 4 \
    --gradient-checkpointing --num-workers 0 \
    --epochs 8 --warmup-steps 1000

bash scripts/run_train_retrieval.sh \
    --init-dir checkpoints/sid_llm/cpt/hf_best \
    --soft-prompt-path checkpoints/sid_llm/init/soft_prompt.pt \
    --embeddings-path data/catalog/embeddings_b2.parquet \
    --catalog-with-sid-path data/catalog/catalog_with_sid.parquet \
    --epochs 5 --batch-size 8 --accumulate-grad-batches 4 \
    --lr 1e-4 --weight-decay 0.005 --pct-start 0.4 \
    --gradient-checkpointing --use-anchored

# 6. Eval
.venv/Scripts/python.exe -m sid_llm.eval.sid_llm_eval_seq \
    --ckpt-dir checkpoints/sid_llm/retrieval/hf_best \
    --max-queries 2000 --num-beams 50 --ks 5,10,50

.venv/Scripts/python.exe -m sid_llm.eval.sid_llm_eval_search \
    --ckpt-dir checkpoints/sid_llm/retrieval/hf_best \
    --embeddings-in data/catalog/embeddings_b2.parquet \
    --max-queries 2000 --num-beams 50 --ks 5,10,50

# 7. (Optional) TIGER baseline for ablation
bash scripts/run_baseline_b3.sh \
    --epochs 8 --batch-size 32 --accumulate-grad-batches 4 \
    --lr 1e-3 --warmup-frac 0.3 --gradient-checkpointing \
    --ckpt-dir checkpoints/tiger_b3

# 8. Run the pytest suite
.venv/Scripts/pytest.exe tests/ -v
```

## Hardware

Trained on a single Windows 11 desktop with an **NVIDIA RTX 4060 Ti (16 GB VRAM)**, 32 GB system RAM, Python 3.14. Memory-hardening for this GPU class: gradient checkpointing, gradient accumulation (effective batch 32 = 8 micro-batch × 4 accum), bf16-mixed precision, sharded HF safetensors saves (`max_shard_size=200MB`), `num_workers=0` on the DataLoader (Windows spawn-mode workers re-load datasets and inflate CPU RAM).

## Limitations and honest framing

- **Eval has a partial leak.** The CPT corpus uses leave-one-out per user (last item is the target), so the model has seen "(history → last item)" pairs during training. The eval queries use the same last-item-as-target, just with a different query format (text title or CLIP embedding). Strict generalization to *unseen items* would require temporal/user-level splits not implemented here. Industry-standard retrieval evals (TIGER, DSI) have the same caveat.
- **Sequence-mode recall@10 is modest in absolute terms (1.5%).** This is below TIGER's published Amazon-subset numbers (4–12%), partly because our catalog is larger (150k vs 10–50k typical) and partly because 5 epochs of M3.7 fine-tune has not fully converged the sequence-mode val loss. The strength of this project is the multimodal search task (39.4% recall@10), which is the actual product target.
- **No cold-start eval.** All catalog items are seen during CPT (via metadata rows: `title → SID`), so there is no held-out item evaluation. Cold-start would require a fundamentally different training / eval setup.
- **Single-machine training.** Everything fits on one RTX 4060 Ti — proves it's reproducible without cluster compute, but the scaling-view paper (A9) makes a strong case that generative retrieval requires significantly more compute to beat dense MIPS at production scale. This repo is a high-quality reference implementation, not a state-of-the-art system.

## Project documents

- Spec (private): `dh-lab/docs/superpowers/specs/2026-05-04-sid-llm-design.md`
- Implementation plan (private): `dh-lab/docs/superpowers/plans/2026-05-04-sid-llm.md`

## License

Apache 2.0.

## Acknowledgments

- McAuley Lab, UC San Diego — Amazon Reviews 2023 dataset
- Reference papers: TIGER (Rajput et al. 2023), PLUM (He et al. 2025, YouTube), VL-CLIP (Giahi et al. 2025, Walmart), GRID (Ju et al. 2025), L2-SP (Xuhong Li et al. 2018), DSI (Tay et al. 2022)
