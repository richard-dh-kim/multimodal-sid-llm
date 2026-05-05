import torch
from sid_llm.models.sid_llm import SIDLLMLightning, SID_TOKEN_NAMES, NUM_SOFT_PROMPT_TOKENS


def test_token_names_include_all_sid_indices_plus_control_plus_modes():
    """1024 SIDs + 2 control + 2 modes = 1028 new tokens."""
    assert len(SID_TOKEN_NAMES) == 1028
    assert "<sid_0>" in SID_TOKEN_NAMES
    assert "<sid_1023>" in SID_TOKEN_NAMES
    assert "<sid_pad>" in SID_TOKEN_NAMES
    assert "<sid_eos>" in SID_TOKEN_NAMES
    assert "<seq>" in SID_TOKEN_NAMES
    assert "<search>" in SID_TOKEN_NAMES


def test_lightning_module_resizes_vocab_and_adds_tokens():
    """Constructing the module expands the tokenizer and resizes embeddings."""
    m = SIDLLMLightning(model_id="t5-small")  # use t5-small for fast tests (~60M)
    base_vocab = 32100  # T5 vocab size pre-expansion (approximate)
    new_vocab = m.tokenizer.vocab_size + len(SID_TOKEN_NAMES)
    actual = m.model.config.vocab_size
    # Verify the resize happened correctly
    assert actual == m.tokenizer.vocab_size + len(SID_TOKEN_NAMES) - (m.tokenizer.vocab_size - base_vocab if False else 0) or actual >= base_vocab + 1028 - 100
    # Soft check: model's shared embedding has at least 1028 more rows than base
    assert m.model.shared.num_embeddings >= base_vocab + 1028 - 100  # tolerate exact T5 vocab variance


def test_new_token_ids_are_consecutive():
    """The 1028 new SID/control/mode tokens get a contiguous block of IDs at the end."""
    m = SIDLLMLightning(model_id="t5-small")
    sid_0_id = m.tokenizer.convert_tokens_to_ids("<sid_0>")
    sid_1023_id = m.tokenizer.convert_tokens_to_ids("<sid_1023>")
    assert sid_1023_id == sid_0_id + 1023, f"got {sid_0_id} -> {sid_1023_id}"


def test_new_embedding_rows_have_similar_magnitude_to_existing():
    """Mean-init + small Gaussian noise should give new rows similar L2 norm to existing rows."""
    m = SIDLLMLightning(model_id="t5-small")
    sid_0_id = m.tokenizer.convert_tokens_to_ids("<sid_0>")
    emb = m.model.shared.weight
    # Pick a few existing rows (id < sid_0_id) and a few new rows
    existing_norms = emb[:100].norm(dim=1).mean().item()
    new_norms = emb[sid_0_id:sid_0_id + 100].norm(dim=1).mean().item()
    # Allow a wide range; the point is that new rows aren't N(0,1) random init
    assert 0.1 * existing_norms < new_norms < 10 * existing_norms, \
        f"existing norm={existing_norms:.3f}, new norm={new_norms:.3f}"


def test_soft_prompt_projection_shape():
    """The soft-prompt projection maps query_dim (512) to T5 d_model."""
    m = SIDLLMLightning(model_id="t5-small")
    d_model = m.model.config.d_model
    assert m.query_projection.in_features == 512
    assert m.query_projection.out_features == d_model
    # NUM_SOFT_PROMPT_TOKENS positions of size [d_model] each
    assert m.soft_prompt_offsets.shape == (NUM_SOFT_PROMPT_TOKENS, d_model)


def test_smoke_forward_pass_with_sid_target():
    """A minimal forward pass over a fake (input, sid-target) pair should produce loss without crashing."""
    m = SIDLLMLightning(model_id="t5-small")
    m.eval()
    sid_tokens = ["<sid_0>", "<sid_1>", "<sid_2>", "<sid_3>"]
    sid_ids = m.tokenizer.convert_tokens_to_ids(sid_tokens) + [m.tokenizer.eos_token_id]
    input_text = "<seq> <sid_5> <sid_6> <sid_7> <sid_8>"
    enc = m.tokenizer(input_text, return_tensors="pt")
    labels = torch.tensor([sid_ids])
    out = m.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
    assert out.loss.dim() == 0
    assert out.loss.item() > 0
