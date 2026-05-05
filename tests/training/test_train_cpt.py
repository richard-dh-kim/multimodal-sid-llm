"""Smoke test for the CPT training step on tiny synthetic data."""
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from sid_llm.training.train_cpt import CPTSeqDataset


def test_dataset_reads_corpus_parquet(tmp_path: Path):
    rows = [
        {"seq_type": "metadata", "input_text": "<seq> title: x", "target_text": "<sid_1><sid_2><sid_eos>"},
        {"seq_type": "behavior", "input_text": "<seq> <sid_3>", "target_text": "<sid_4><sid_eos>"},
    ]
    p = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(p))
    ds = CPTSeqDataset(p)
    assert len(ds) == 2
    assert ds[0]["input_text"] == "<seq> title: x"
    assert ds[1]["target_text"] == "<sid_4><sid_eos>"
