"""AdamW with L2-SP: weight decay pulls toward captured init weights, not zero."""
from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim.optimizer import Optimizer


class AdamWAnchored(Optimizer):
    """AdamW that decays toward init weights (L2-SP) instead of toward zero.

    Anchor `theta_init` is a detached clone captured per-parameter at __init__,
    living on the parameter's own device. Update rule:
        theta -= lr * m_hat / (sqrt(v_hat) + eps) + lr * wd * (theta - theta_init)
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
                        # add_param_group path: anchor on first sight.
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

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if wd != 0.0:
                    p.add_(p - theta_init, alpha=-lr * wd)

        return loss
