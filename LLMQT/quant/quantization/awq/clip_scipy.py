"""SciPy refinement for AWQ per-output-channel/group clipping ratios.

The regular AWQ grid search remains the source of the initial clipping ratio.
This module only refines that result continuously.  The caller supplies a
differentiable STE objective and an exact round/clamp error evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from scipy.optimize import minimize


@dataclass
class ScipyClipRefinement:
    ratios: torch.Tensor
    accepted_groups: int
    total_groups: int
    initial_loss: float
    final_loss: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    max_threshold_change: float
    callback_triggered: bool
    callback_reason: Optional[str]

    @property
    def relative_improvement(self) -> float:
        if self.initial_loss <= 0:
            return 0.0
        return (self.initial_loss - self.final_loss) / self.initial_loss

    def diagnostics(self) -> dict:
        return {
            "accepted_groups": self.accepted_groups,
            "total_groups": self.total_groups,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "relative_improvement": self.relative_improvement,
            "optimizer_success": self.optimizer_success,
            "optimizer_status": self.optimizer_status,
            "optimizer_message": self.optimizer_message,
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "gradient_evaluations": self.gradient_evaluations,
            "max_threshold_change": self.max_threshold_change,
            "callback_triggered": self.callback_triggered,
            "callback_reason": self.callback_reason,
        }


def refine_clip_ratios(
    initial_ratios: torch.Tensor,
    initial_errors: torch.Tensor,
    differentiable_objective: Callable[[torch.Tensor], torch.Tensor],
    exact_errors: Callable[[torch.Tensor], torch.Tensor],
    *,
    maxiter: int = 10_000_000,
    min_ratio: float = 0.5,
    max_ratio: float = 1.0,
    max_threshold_change: float = 0.10,
) -> ScipyClipRefinement:
    """Jointly refine one output-channel batch of AWQ clipping ratios.

    A callback stops the optimizer when any threshold changes by more than
    ``max_threshold_change`` relative to its grid-search initial value.  That
    event rolls the complete batch back to the grid result.  Otherwise, exact
    round/clamp validation accepts only the individual groups whose MSE falls.
    """

    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if not 0.0 < min_ratio < max_ratio <= 1.0:
        raise ValueError("clip ratio bounds must satisfy 0 < min < max <= 1")
    if not 0.0 < max_threshold_change < 1.0:
        raise ValueError("max_threshold_change must be between 0 and 1")
    if initial_ratios.shape != initial_errors.shape:
        raise ValueError("initial_ratios and initial_errors must have equal shapes")

    device = initial_ratios.device
    shape = initial_ratios.shape
    initial_ratios = initial_ratios.detach().float().clamp(min_ratio, max_ratio)
    initial_errors = initial_errors.detach().float()
    x0 = initial_ratios.cpu().numpy().astype(np.float64).reshape(-1)
    bounds = [(min_ratio, max_ratio)] * x0.size

    best = {"loss": float("inf"), "x": x0.copy()}
    callback_state = {"triggered": False, "reason": None}

    def relative_threshold_change(candidate: np.ndarray) -> float:
        denominator = np.maximum(np.abs(x0), np.finfo(np.float64).eps)
        return float(np.max(np.abs(candidate - x0) / denominator))

    def objective(candidate: np.ndarray):
        with torch.enable_grad():
            # The caller is normally wrapped in torch.no_grad() because the
            # original AWQ grid search is inference-only. Create and reshape
            # the leaf inside enable_grad() so the SciPy Jacobian is retained.
            ratios = torch.tensor(
                candidate.reshape(shape),
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )
            loss = differentiable_objective(ratios)
        if not torch.isfinite(loss):
            return np.finfo(np.float64).max / 1024.0, np.zeros_like(candidate)
        loss.backward()
        if ratios.grad is None:
            raise RuntimeError("AWQ SciPy clip objective did not produce a gradient")
        loss_value = float(loss.detach().cpu())
        gradient = ratios.grad.detach().cpu().numpy().astype(np.float64).reshape(-1)
        if loss_value < best["loss"]:
            best["loss"] = loss_value
            best["x"] = np.asarray(candidate, dtype=np.float64).copy()
        return loss_value, gradient

    def rollback_callback(candidate: np.ndarray):
        change = relative_threshold_change(np.asarray(candidate))
        if change > max_threshold_change:
            callback_state["triggered"] = True
            callback_state["reason"] = (
                f"max clip-threshold change {change:.6%} exceeded the "
                f"{max_threshold_change:.6%} rollback threshold"
            )
            raise StopIteration

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=rollback_callback,
        options={"maxiter": int(maxiter)},
    )

    # L-BFGS-B may terminate during line search without invoking its callback.
    # Apply the same guard to the best evaluated point before exact acceptance.
    candidate_np = best["x"] if np.isfinite(best["loss"]) else result.x
    candidate_change = relative_threshold_change(candidate_np)
    if (
        not callback_state["triggered"]
        and candidate_change > max_threshold_change
    ):
        callback_state["triggered"] = True
        callback_state["reason"] = (
            f"max clip-threshold change {candidate_change:.6%} exceeded the "
            f"{max_threshold_change:.6%} rollback threshold"
        )

    if callback_state["triggered"]:
        final_ratios = initial_ratios
        final_errors = initial_errors
        accepted = torch.zeros_like(initial_errors, dtype=torch.bool)
        measured_change = candidate_change
    else:
        candidate_ratios = torch.from_numpy(candidate_np).to(
            device=device, dtype=torch.float32
        ).view(shape)
        candidate_errors = exact_errors(candidate_ratios).detach().float()
        accepted = torch.isfinite(candidate_errors) & (candidate_errors < initial_errors)
        final_ratios = torch.where(accepted, candidate_ratios, initial_ratios)
        final_errors = torch.where(accepted, candidate_errors, initial_errors)
        measured_change = relative_threshold_change(candidate_np)

    return ScipyClipRefinement(
        ratios=final_ratios.detach(),
        accepted_groups=int(accepted.sum().item()),
        total_groups=initial_ratios.numel(),
        initial_loss=float(initial_errors.mean().item()),
        final_loss=float(final_errors.mean().item()),
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        function_evaluations=int(getattr(result, "nfev", 0)),
        gradient_evaluations=int(getattr(result, "njev", 0)),
        max_threshold_change=measured_change,
        callback_triggered=bool(callback_state["triggered"]),
        callback_reason=callback_state["reason"],
    )


def _demo_fake_quantize(w: torch.Tensor) -> torch.Tensor:
    shape = w.shape
    grouped = w.reshape(-1, shape[-1])
    max_val = grouped.amax(dim=1, keepdim=True)
    min_val = grouped.amin(dim=1, keepdim=True)
    scale = (max_val - min_val).clamp_min(1e-5) / 15
    zero = (-torch.round(min_val / scale)).clamp(0, 15)
    quantized = (
        torch.clamp(torch.round(grouped / scale) + zero, 0, 15) - zero
    ) * scale
    return quantized.reshape(shape)


def demo() -> None:
    """Run a tiny deterministic W4 clipping refinement example on CPU."""

    torch.manual_seed(0)
    weight = torch.randn(4, 1, 2, 16)
    weight[:, :, :, 0] *= 5
    inputs = torch.randn(1, 96, 2, 16)
    reference = (inputs * weight).sum(dim=-1)
    original_max = weight.abs().amax(dim=-1, keepdim=True)

    def errors(ratios: torch.Tensor, use_ste: bool = False) -> torch.Tensor:
        threshold = original_max * ratios
        clipped = torch.clamp(weight, -threshold, threshold)
        quantized = _demo_fake_quantize(clipped)
        if use_ste:
            quantized = clipped + (quantized - clipped).detach()
        output = (inputs * quantized).sum(dim=-1)
        return (output - reference).float().pow(2).mean(dim=1).view(
            original_max.shape
        )

    grid_errors = torch.stack(
        [errors(torch.full_like(original_max, 1 - index / 20)) for index in range(10)]
    )
    initial_errors, best_indices = grid_errors.min(dim=0)
    initial_ratios = 1 - best_indices.float() / 20
    result = refine_clip_ratios(
        initial_ratios,
        initial_errors,
        differentiable_objective=lambda ratios: errors(
            ratios, use_ste=True
        ).mean(),
        exact_errors=errors,
        maxiter=10_000_000,
        max_threshold_change=0.10,
    )

    print("Synthetic AWQ SciPy clip refinement")
    print(f"grid mean MSE       : {result.initial_loss:.8f}")
    print(f"refined mean MSE    : {result.final_loss:.8f}")
    print(f"relative improvement: {result.relative_improvement * 100:.4f}%")
    print(f"accepted groups     : {result.accepted_groups}/{result.total_groups}")
    print(f"rollback callback   : {result.callback_triggered}")
    print(f"iterations/evals    : {result.iterations}/{result.function_evaluations}")


if __name__ == "__main__":
    demo()
