from PIL import Image
import torch
from sid_llm.models.clip_embedder import ClipEmbedder


def test_clip_embedder_returns_512_dim_vector():
    embedder = ClipEmbedder(device="cpu")
    img = Image.new("RGB", (224, 224), "red")
    emb = embedder.embed_image_text(img, "red square")
    assert emb.shape == (512,)


def test_clip_embedder_batch_returns_correct_shape():
    embedder = ClipEmbedder(device="cpu")
    imgs = [Image.new("RGB", (224, 224), "red") for _ in range(3)]
    texts = ["red"] * 3
    embs = embedder.embed_image_text_batch(imgs, texts)
    assert embs.shape == (3, 512)


def test_clip_embedder_l2_normalizes_output():
    embedder = ClipEmbedder(device="cpu")
    img = Image.new("RGB", (224, 224), "blue")
    emb = embedder.embed_image_text(img, "blue square")
    assert abs(torch.norm(emb).item() - 1.0) < 1e-3
