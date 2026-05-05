# multimodal-sid-llm

Multimodal generative retrieval over Semantic IDs. The model takes a query (text or photo) and returns matching items from a ~150k Amazon Products catalog by autoregressively generating the items' discrete 4-token semantic IDs.

> 🚧 **Active development.** M0 (data pipeline) verified at production scale; M1 (preprocessing + plain-CLIP baseline) in progress. Public release targeted after M5.

## Architecture

```
   query (text or image)
          │
          ▼
  ┌──────────────────────────────────┐
  │  Visual + text encoder           │
  │  (CLIP ViT-B/32, fine-tuned)     │
  └──────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────┐
  │  Soft-prompt projection          │
  │  query embedding → T5 input      │
  └──────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────┐
  │  T5-base, vocabulary expanded    │
  │  with 1024 SID tokens            │
  │  + continued pre-training        │
  │  + generative retrieval finetune │
  └──────────────────────────────────┘
          │
          ▼
   beam search (4 steps × K beams)
   with KV cache + Trie-constrained decoding
          │
          ▼
   {(t1,t2,t3,t4): item_id} lookup
          │
          ▼
   top-K items
```

## Distinguishing contributions vs prior art

- Multimodal front-end (VL-CLIP-style cropping + LLM-augmented text — eventually) feeding a TIGER/PLUM-style SID-LLM. Most published SID-LLM work is item-ID only.
- KV caching during beam search — a known gap in production reference codebases (~32× speedup on the decode loop).
- Trie-constrained decoding — guarantees zero hallucinated SIDs.

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
├── configs/            ← Hydra YAML configs (M2+)
├── data/               ← gitignored; rebuilt by download scripts
└── checkpoints/        ← gitignored; produced by training
```

## Reproduction

```bash
# 1. Setup
py -m venv .venv
.venv/Scripts/pip install -e ".[dev]"

# 2. Run the pipeline
bash scripts/run_download_metadata.sh    # ~36 sec
bash scripts/run_download_reviews.sh     # ~14 min
bash scripts/run_download_images.sh      # ~18 min (99.96% success rate)
bash scripts/run_build_catalog.sh        # ~30 sec

# 3. Run tests
.venv/Scripts/pytest.exe tests/ -v
```

After the data pipeline finishes, `data/catalog/catalog.parquet` exists with 150k rows. Detailed usage for the modeling stages will be documented as M2+ ships.

## License

To be determined before public release.

## Acknowledgments

- McAuley Lab, UC San Diego — Amazon Reviews 2023 dataset
- Reference papers: TIGER (Rajput et al. 2023), PLUM (He et al. 2025, YouTube), VL-CLIP (Giahi et al. 2025, Walmart), GRID (Ju et al. 2025)
