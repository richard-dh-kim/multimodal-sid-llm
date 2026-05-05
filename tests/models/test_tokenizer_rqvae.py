import torch
from sid_llm.models.tokenizer_rqvae import RQVAETokenizer


def _seeded_tokenizer(num_quantizers=4, codebook_size=1024, dim=64, seed=0):
    """Small RQ-VAE for fast tests. dim=64 keeps tests under a second on CPU."""
    torch.manual_seed(seed)
    return RQVAETokenizer(
        dim=dim,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        commitment_weight=0.25,
        kmeans_init=False,  # speed up tests
    )


def test_forward_returns_4_indices_per_item():
    torch.manual_seed(0)
    tok = _seeded_tokenizer(num_quantizers=4, codebook_size=1024, dim=64)
    x = torch.randn(8, 64)
    quantized, indices, commit_loss = tok(x)
    assert quantized.shape == (8, 64)
    assert indices.shape == (8, 4)
    assert commit_loss.dim() == 0  # scalar


def test_indices_are_in_codebook_range():
    torch.manual_seed(1)
    tok = _seeded_tokenizer(num_quantizers=4, codebook_size=1024, dim=64)
    x = torch.randn(32, 64)
    _, indices, _ = tok(x)
    assert (indices >= 0).all()
    assert (indices < 1024).all()


def test_commitment_loss_nonnegative():
    torch.manual_seed(2)
    tok = _seeded_tokenizer()
    x = torch.randn(16, 64)
    _, _, commit_loss = tok(x)
    assert commit_loss.item() >= 0.0


def test_eval_mode_is_deterministic():
    """In eval mode the same input must produce the same indices (no codebook updates)."""
    torch.manual_seed(3)
    tok = _seeded_tokenizer()
    tok.eval()
    x = torch.randn(8, 64)
    _, indices_a, _ = tok(x)
    _, indices_b, _ = tok(x)
    assert torch.equal(indices_a, indices_b)


def test_default_dim_512_for_clip_embeds():
    """Default dim should match CLIP ViT-B/32 output (512)."""
    tok = RQVAETokenizer()
    x = torch.randn(2, 512)
    quantized, indices, _ = tok(x)
    assert quantized.shape == (2, 512)
    assert indices.shape == (2, 4)
