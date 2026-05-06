"""AdamW variants for SID-LLM training.

`AdamWAnchored` implements L2-SP regularization: the weight-decay term pulls
parameters toward a captured `theta_init` rather than toward zero. This keeps
the fine-tune from forgetting CPT-learned representations.

Update rule (per parameter, ignoring bias correction details, identical to
torch.optim.AdamW except for the decay target):

    m_t   = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t   = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    theta -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * (theta - theta_init))

`theta_init` is captured (cloned, detached, on the same device/dtype) at
optimizer construction time and never updated.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim.optimizer import Optimizer


class AdamWAnchored(Optimizer):
    """AdamW that decays toward init weights (L2-SP) instead of toward zero.

    Same constructor surface as torch.optim.AdamW. Captures a frozen clone of
    every parameter at __init__ time as the anchor for that parameter.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # Snapshot the anchor (theta_init) for every parameter under management.
        # Stored on the parameter's own device so the decay term never crosses
        # devices on the hot path.
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["theta_init"] = p.detach().clone()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamWAnchored does not support sparse gradients.")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if "theta_init" not in state:
                        # If a parameter was added to the optimizer after __init__
                        # (rare; e.g. add_param_group), anchor it on first sight.
                        state["theta_init"] = p.detach().clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                theta_init = state["theta_init"]

                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1

                # Adam step.
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # L2-SP anchor decay: theta -= lr * wd * (theta - theta_init)
                if wd != 0.0:
                    p.add_(p - theta_init, alpha=-lr * wd)

        return loss
