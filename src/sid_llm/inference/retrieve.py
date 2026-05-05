"""Convenience re-exports for the SID-LLM beam-search retrieval API.

The implementation lives in `beam_search.py`; this module exposes the public
surface (`BeamSearchRetriever`, `load_retriever`) under a stable name so callers
can write `from sid_llm.inference.retrieve import load_retriever`.
"""
from __future__ import annotations

from sid_llm.inference.beam_search import BeamSearchRetriever, load_retriever

__all__ = ["BeamSearchRetriever", "load_retriever"]
