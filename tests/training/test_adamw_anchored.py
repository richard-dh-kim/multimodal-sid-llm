"""Tests for AdamWAnchored: L2-SP decay toward init weights, not zero."""
from __future__ import annotations

import torch

from sid_llm.training.optimizers import AdamWAnchored


def test_decays_toward_init():
    torch.manual_seed(0)
    init_value = 5.0
    p = torch.nn.Parameter(torch.full((4,), init_value))
    opt = AdamWAnchored([p], lr=1e-1, weight_decay=1e-2)

    # Perturb after the optimizer captures the anchor.
    with torch.no_grad():
        p.add_(torch.tensor([1.0, -1.0, 2.0, -2.0]))

    perturbed = p.detach().clone()
    init_anchor = opt.state[p]["theta_init"]
    assert torch.equal(init_anchor, torch.full((4,), init_value)), \
        "Anchor must equal init value, not the perturbed value."

    for _ in range(50):
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        else:
            p.grad.zero_()
        opt.step()

    d_init_before = (perturbed - init_anchor).abs().sum().item()
    d_init_after = (p.detach() - init_anchor).abs().sum().item()
    d_zero_before = perturbed.abs().sum().item()
    d_zero_after = p.detach().abs().sum().item()

    assert d_init_after < d_init_before, (
        f"Weights should drift toward theta_init. "
        f"d_init_before={d_init_before:.4f} d_init_after={d_init_after:.4f}"
    )
    # Vanilla AdamW would shrink d_zero; AdamWAnchored should not.
    assert d_zero_after >= d_zero_before * 0.95, (
        f"Weights should NOT decay toward zero. "
        f"d_zero_before={d_zero_before:.4f} d_zero_after={d_zero_after:.4f}"
    )

    # Init=5, perturbed=[6,4,7,3]; all should stay positive and approach 5.
    expected_signs = torch.tensor([1.0, 1.0, 1.0, 1.0])
    actual_signs = torch.sign(p.detach())
    assert torch.equal(actual_signs, expected_signs), \
        f"Sign flipped under anchor decay (would imply over-shoot toward zero): {p.detach()}"


def test_step_with_gradient_does_descent():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    opt = AdamWAnchored([p], lr=1e-1, weight_decay=0.0)
    target = torch.tensor([0.0, 0.0])

    for _ in range(100):
        opt.zero_grad()
        loss = ((p - target) ** 2).sum()
        loss.backward()
        opt.step()

    assert p.detach().abs().sum().item() < 0.5, \
        f"Adam term failed to descend toward target: p={p.detach()}"


def test_anchor_captured_before_first_step():
    p = torch.nn.Parameter(torch.tensor([3.0, 3.0]))
    opt = AdamWAnchored([p], lr=1e-2)
    captured = opt.state[p]["theta_init"]
    with torch.no_grad():
        p.fill_(99.0)
    assert torch.equal(captured, torch.tensor([3.0, 3.0]))
