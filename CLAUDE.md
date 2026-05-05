# multimodal-sid-llm — Code Repo

This is the **public** repository for the multimodal SID-LLM project. Code, tests, and demo only. Brainstorming notes, specs, and implementation plans live in the **private** sibling repo `dh-lab` (https://github.com/richard-dh-kim/dh-lab) — that's where the design context, paper bibliography, and progress logs live.

GitHub: https://github.com/richard-dh-kim/multimodal-sid-llm

---

## What this project is

Multimodal generative retrieval over Semantic IDs. A user provides a query (text or photo) and the system returns matching items from a ~150k Amazon Products catalog by having an LLM autoregressively generate the items' discrete semantic IDs. Three-stage pipeline: visual front-end (cropped CLIP image embedding + cleaned text embedding) → SID tokenizer (RQ-VAE / RQ-KMeans into 4 codes × 1024 vocab) → small LLM with SID-extended vocabulary trained for generative retrieval.

For the design rationale, paper references, and full implementation plan: see the dh-lab repo's `docs/superpowers/specs/2026-05-04-sid-llm-design.md` and `docs/superpowers/plans/2026-05-04-sid-llm.md`.

---

## Working with this repo

### Setup (per-machine)

```bash
py -m venv .venv
.venv/Scripts/pip install -e ".[dev]"   # Windows
# or .venv/bin/pip install -e ".[dev]"  # POSIX
```

### Commands

| What | How |
|---|---|
| Run all tests | `.venv/Scripts/pytest.exe tests/ -v` |
| Run one test file | `.venv/Scripts/pytest.exe tests/data/test_X.py -v` |
| Lint | `.venv/Scripts/ruff.exe check src/ tests/` |
| Download metadata | `bash scripts/run_download_metadata.sh` |
| Download reviews | `bash scripts/run_download_reviews.sh` |
| Download images | `bash scripts/run_download_images.sh` |
| Build catalog | `bash scripts/run_build_catalog.sh` |

### Layout

```
multimodal-sid-llm/
├── src/sid_llm/
│   ├── data/                ← download + preprocessing modules
│   ├── models/              ← embedder, tokenizer, SID-LLM model wrappers
│   ├── training/            ← (M2+) Lightning modules and CLI entry points
│   ├── inference/           ← (M3+) beam search, Trie-constrained decoding
│   ├── eval/                ← metrics + baselines
│   └── demo/                ← (M5) Gradio app
├── tests/                   ← pytest, mirrors src/sid_llm/ structure
├── scripts/                 ← shell entry points per pipeline stage
├── configs/                 ← (M2+) Hydra YAML configs
├── data/                    ← gitignored; rebuilt by scripts
└── checkpoints/             ← gitignored; produced by training
```

### Conventions

- **Python 3.11+**, type hints on public functions, click for CLIs.
- **TDD where the function has clear input→output behavior.** Pure functions (filters, path derivations, metric calcs) get unit tests with inline dicts; classes that load real models (CLIP, T5, Grounding DINO) are exercised via integration smoke tests, not mocks.
- **Streaming approach** for any large dataset — never download wholesale; read line-by-line via `urllib.request.urlopen` + `gzip.GzipFile`.
- **Parquet for tabular outputs** (snappy compression). One row per item or per interaction.
- **One module = one responsibility.** Don't extract shared helpers prematurely; the rule of three applies.
- **Tests must verify behavior, not implementation.** No mocking what's already cheap to run; no asserting on private state.

### Subagent dispatching

When the agent dispatches subagents (Agent tool) for code or doc work on this repo, use the `opus` model. Cost is acceptable; quality is the priority.

### Git commit attribution

Commits in this repo are authored solely by Richard. Do not add `Co-Authored-By: Claude`, `Generated with Claude Code`, or any other Claude-related attribution to commit messages or PR descriptions.

---

## Cross-repo workflow

This repo is meant to be cloned alongside the private `dh-lab` repo at the same parent directory:

```
~/Desktop/career/
├── dh-lab/                  ← private workspace (specs, plans, brainstorming)
└── multimodal-sid-llm/      ← this repo (code)
```

For the design spec, implementation plan, paper references, and project decomposition story, look at `../dh-lab/docs/superpowers/`, `../dh-lab/notes/`, and `../dh-lab/research/`.

When working on this repo with Claude Code, the `dh-lab/CLAUDE.md` is the canonical context source for the broader project. This file (the one you're reading) is scoped to code-level concerns within this repo.

---

## Status

| Milestone | State |
|---|---|
| M0 — Data pipeline | ✅ Done (verified at production scale: 150k items, 11.8M interactions, 99.96% images) |
| M1 — Preprocessing + plain CLIP baseline | 🟡 In progress. M1.1 (Grounding DINO module) ✅, M1.3 (text cleaner) ✅, M1.4 partial (CLIP embedder) ✅. M1.2 (Grounding DINO orchestration) deferred — see spec. |
| M2 — VL-CLIP fine-tune + B2 | ⏸ Not started |
| M3 — SID tokenizer + base SID-LLM | ⏸ Not started |
| M4 — Constrained decoding + ablations | ⏸ Not started |
| M5 — Demo + HSTU baseline + write-up | ⏸ Not started |

Update this table as milestones advance.
