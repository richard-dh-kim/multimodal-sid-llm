"""CLI: train the RQ-VAE tokenizer on item embeddings, save SIDs to item2feat.parquet."""
from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sid_llm.models.tokenizer_rqvae import RQVAETokenizer


def _quantize_all(
    model: RQVAETokenizer, embeddings: np.ndarray, batch_size: int, device: str
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            x = torch.from_numpy(embeddings[i:i + batch_size]).to(device)
            _, idx, _ = model(x)
            out.append(idx.cpu().numpy())
    return np.concatenate(out, axis=0)


@click.command()
@click.option(
    "--embeddings-in", default="data/catalog/embeddings_b1.parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--ckpt-out", default="checkpoints/tokenizer/rqvae.pt",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--item2feat-out", default="data/catalog/item2feat.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--epochs", default=20, type=int)
@click.option("--batch-size", default=512, type=int)
@click.option("--lr", default=1e-3, type=float)
@click.option("--num-quantizers", default=4, type=int)
@click.option("--codebook-size", default=1024, type=int)
@click.option("--commitment-weight", default=0.25, type=float)
@click.option("--device", default="cuda", type=str)
@click.option("--seed", default=42, type=int)
def main(
    embeddings_in, ckpt_out, item2feat_out,
    epochs, batch_size, lr, num_quantizers, codebook_size, commitment_weight,
    device, seed,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    item2feat_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {embeddings_in} ...")
    table = pq.read_table(str(embeddings_in))
    item_ids = np.array(table.column("item_id").to_pylist(), dtype=np.int64)
    embeddings = np.array(table.column("embedding").to_pylist(), dtype=np.float32)
    n, dim = embeddings.shape
    print(f"  {n:,} items x {dim}-D")

    if not torch.cuda.is_available() and device == "cuda":
        print("  CUDA unavailable; falling back to CPU.")
        device = "cpu"

    model = RQVAETokenizer(
        dim=dim,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        commitment_weight=commitment_weight,
        kmeans_init=True,
    ).to(device)

    # ResidualVQ updates codebooks via EMA on forward; if there are no learnable
    # parameters we skip the optimizer entirely. Otherwise we run AdamW.
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(learnable_params, lr=lr) if learnable_params else None
    if optimizer is None:
        print("  No learnable parameters; codebooks will be updated via EMA on forward.")

    ds = TensorDataset(torch.from_numpy(embeddings))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"\nTraining RQ-VAE: {num_quantizers} codebooks x {codebook_size} codes, "
          f"epochs={epochs}, batch={batch_size}, lr={lr}, device={device}")
    start = time.time()
    for epoch in range(epochs):
        model.train()
        recon_sum = commit_sum = 0.0
        steps = 0
        for (batch,) in loader:
            batch = batch.to(device)
            quantized, _, commit_loss = model(batch)
            recon_loss = F.mse_loss(quantized, batch)
            loss = recon_loss + commit_loss
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # else: codebooks updated in-place via EMA during forward()
            recon_sum += recon_loss.item()
            commit_sum += commit_loss.item()
            steps += 1
        print(f"  epoch {epoch + 1:2d}/{epochs}  recon={recon_sum / steps:.4f}  "
              f"commit={commit_sum / steps:.4f}  ({time.time() - start:.0f}s)")

    print(f"\nQuantizing all {n:,} items ...")
    sids = _quantize_all(model, embeddings, batch_size=batch_size, device=device)
    print(f"  sids.shape={sids.shape}, dtype={sids.dtype}")

    # Codebook usage diagnostics
    print("\nCodebook usage (unique codes used per level):")
    for level in range(num_quantizers):
        used = len(np.unique(sids[:, level]))
        print(f"  level {level}: {used:,} / {codebook_size} = {used / codebook_size:.1%}")

    # Collision check
    sid_tuples = [tuple(int(c) for c in row) for row in sids]
    unique_tuples = len(set(sid_tuples))
    collisions = n - unique_tuples
    print(f"\nUnique SID tuples: {unique_tuples:,} / {n:,} (collisions = {collisions:,})")

    # Save tokenizer checkpoint
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "dim": dim,
                "num_quantizers": num_quantizers,
                "codebook_size": codebook_size,
                "commitment_weight": commitment_weight,
            },
        },
        str(ckpt_out),
    )
    print(f"\nSaved tokenizer ckpt -> {ckpt_out}")

    # Save item2feat parquet
    table_out = pa.Table.from_pydict({
        "item_id": item_ids.tolist(),
        "sid_0": sids[:, 0].tolist(),
        "sid_1": sids[:, 1].tolist(),
        "sid_2": sids[:, 2].tolist(),
        "sid_3": sids[:, 3].tolist(),
    } if num_quantizers == 4 else {
        "item_id": item_ids.tolist(),
        **{f"sid_{i}": sids[:, i].tolist() for i in range(num_quantizers)},
    })
    pq.write_table(table_out, str(item2feat_out), compression="snappy")
    print(f"Saved item2feat -> {item2feat_out}")


if __name__ == "__main__":
    main()
