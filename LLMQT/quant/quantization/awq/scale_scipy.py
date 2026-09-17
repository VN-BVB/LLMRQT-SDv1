"""SciPy refinement for AWQ per-input-channel equalization scales.

The ordinary AWQ search generates a complete scale vector from one scalar
``ratio`` and evaluates a small grid of ratio values. This module starts from
the best grid-search vector and lets L-BFGS-B adjust every input channel.

Quantization contains ``round`` and ``clamp`` and is therefore not smoothly
differentiable. The caller supplies a PyTorch objective whose forward pass uses
the real fake-quantized weights and whose backward pass uses a straight-through
estimator (STE). SciPy performs the nonlinear optimization; PyTorch computes
the vectorized objective and approximate gradient. A second, non-gradient
objective validates the candidate with the original AWQ forward path. The grid
result is retained unless that exact loss improves.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from scipy.optimize import minimize


TensorObjective = Callable[[torch.Tensor], torch.Tensor]
ExactObjective = Callable[[torch.Tensor], float]


@dataclass
class ScipyScaleRefinement:
    """Serializable diagnostics plus the accepted scale tensor."""

    scales: torch.Tensor
    accepted: bool
    initial_loss: float
    candidate_loss: float
    relative_improvement: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    input_channels: int
    relative_bound: float
    max_scale_change: float
    callback_triggered: bool
    callback_reason: str | None

    def diagnostics(self) -> dict:
        payload = asdict(self)
        payload.pop("scales")
        return payload


def channel_scales_from_log_correction(
    initial_scales: torch.Tensor,
    log_correction: torch.Tensor,
) -> torch.Tensor:
    """Apply one multiplicative correction per channel.

    Subtracting the mean removes the unidentifiable global scale direction:
    multiplying every AWQ scale by the same constant is cancelled by the
    corresponding inverse activation scaling (up to floating-point effects).
    At ``log_correction == 0`` this returns ``initial_scales`` exactly.
    """

    centered = log_correction - log_correction.mean()
    return initial_scales * centered.exp()


def refine_channel_scales(
    initial_scales: torch.Tensor,
    initial_loss: float,
    differentiable_objective: TensorObjective,
    exact_objective: ExactObjective,
    *,
    maxiter: int = 10_000_000,
    relative_bound: float = 1.25,
    max_scale_change: float = 0.10,
    ftol: float = 1e-9,
    gtol: float = 1e-5,
    maxls: int = 8,
    acceptance_rtol: float = 1e-7,
) -> ScipyScaleRefinement:
    """Refine all input-channel scales with bounded SciPy L-BFGS-B.

    ``differentiable_objective`` receives a tensor containing log-scale
    corrections and returns a scalar loss. It must provide an STE or another
    useful gradient through fake quantization. ``exact_objective`` receives a
    concrete scale vector and evaluates the original non-differentiable AWQ
    objective. Only an exact improvement is accepted.
    """

    if initial_scales.ndim != 1:
        raise ValueError("initial_scales must be a one-dimensional tensor")
    if not math.isfinite(initial_loss):
        raise ValueError("initial_loss must be finite")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")
    if relative_bound <= 1.0:
        raise ValueError("relative_bound must be greater than 1")
    if not 0.0 < max_scale_change < 1.0:
        raise ValueError("max_scale_change must be between 0 and 1")

    initial_scales = initial_scales.detach()
    scipy_dtype = np.float64
    x0 = np.zeros(initial_scales.numel(), dtype=scipy_dtype)
    log_bound = math.log(relative_bound)
    bounds = [(-log_bound, log_bound)] * initial_scales.numel()

    # SciPy may finish on a line-search point other than the best point it
    # evaluated. Preserve the lowest real forward loss seen by the optimizer.
    best = {"loss": float(initial_loss), "x": x0.copy()}
    callback_state = {
        "triggered": False,
        "reason": None,
    }

    def relative_scale_change(log_correction_np: np.ndarray) -> float:
        centered = log_correction_np - np.mean(log_correction_np)
        correction = np.exp(centered)
        return float(np.max(np.abs(correction - 1.0)))

    def rollback_callback(log_correction_np: np.ndarray) -> None:
        """Stop this mapping when an accepted iterate moves over 10%."""

        change = relative_scale_change(np.asarray(log_correction_np))
        if change > max_scale_change:
            callback_state["triggered"] = True
            callback_state["reason"] = (
                f"max per-channel scale change {change:.6%} exceeded "
                f"the {max_scale_change:.6%} rollback threshold"
            )
            # SciPy converts StopIteration raised by a callback into an early
            # termination result. The selection logic below then restores the
            # original grid-search scale vector.
            raise StopIteration(callback_state["reason"])

    def value_and_grad(log_correction_np: np.ndarray):
        log_correction = torch.tensor(
            log_correction_np,
            device=initial_scales.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        with torch.enable_grad():
            loss = differentiable_objective(log_correction)

        if loss.ndim != 0:
            raise ValueError("differentiable_objective must return a scalar")

        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            return np.finfo(scipy_dtype).max / 1024.0, np.zeros_like(x0)

        loss.backward()
        gradient = log_correction.grad
        if gradient is None:
            raise RuntimeError("AWQ SciPy objective did not produce a gradient")
        gradient_np = gradient.detach().float().cpu().numpy().astype(scipy_dtype)
        if not np.all(np.isfinite(gradient_np)):
            gradient_np = np.nan_to_num(
                gradient_np, nan=0.0, posinf=0.0, neginf=0.0
            )

        if loss_value < best["loss"]:
            best["loss"] = loss_value
            best["x"] = np.asarray(log_correction_np, dtype=scipy_dtype).copy()
        return loss_value, gradient_np

    result = minimize(
        value_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=rollback_callback,
        options={
            "maxiter": maxiter,
            "maxls": maxls,
            "ftol": ftol,
            "gtol": gtol,
            "maxcor": min(10, maxiter),
        },
    )

    best_log_correction = torch.tensor(
        best["x"],
        device=initial_scales.device,
        dtype=torch.float32,
    )
    candidate_scales = channel_scales_from_log_correction(
        initial_scales.float(), best_log_correction
    ).to(initial_scales.dtype)
    candidate_loss = float(exact_objective(candidate_scales))

    improvement = float(initial_loss) - candidate_loss
    required_improvement = abs(float(initial_loss)) * acceptance_rtol
    accepted = (
        not callback_state["triggered"]
        and math.isfinite(candidate_loss)
        and improvement > required_improvement
    )
    selected_scales = candidate_scales if accepted else initial_scales
    relative_improvement = improvement / max(abs(float(initial_loss)), 1e-30)

    return ScipyScaleRefinement(
        scales=selected_scales.detach(),
        accepted=accepted,
        initial_loss=float(initial_loss),
        candidate_loss=candidate_loss,
        relative_improvement=relative_improvement,
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        gradient_evaluations=int(result.njev),
        input_channels=int(initial_scales.numel()),
        relative_bound=float(relative_bound),
        max_scale_change=float(max_scale_change),
        callback_triggered=bool(callback_state["triggered"]),
        callback_reason=callback_state["reason"],
    )


def _fake_quantize_asymmetric_groupwise(
    weight: torch.Tensor,
    group_size: int,
    bits: int = 4,
    ste: bool = False,
) -> torch.Tensor:
    """Small self-contained fake quantizer used only by the demo."""

    original_shape = weight.shape
    grouped = weight.reshape(-1, group_size)
    minimum = grouped.amin(dim=1, keepdim=True)
    maximum = grouped.amax(dim=1, keepdim=True)
    qmax = 2**bits - 1
    scale = (maximum - minimum).clamp(min=1e-5) / qmax
    zero = (-torch.round(minimum / scale)).clamp(0, qmax)
    quantized = (
        torch.clamp(torch.round(grouped / scale) + zero, 0, qmax) - zero
    ) * scale
    quantized = quantized.reshape(original_shape)
    if ste:
        return weight + (quantized - weight).detach()
    return quantized


def run_demo(
    seed: int = 7,
    maxiter: int = 10_000_000,
    max_scale_change: float = 0.10,
) -> ScipyScaleRefinement:
    """Run a deterministic synthetic AWQ refinement without loading an LLM."""

    torch.manual_seed(seed)
    samples, in_features, out_features, group_size = 192, 32, 48, 8
    weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.15
    # Deliberately create channel/group outliers so per-channel equalization has
    # room to improve beyond the one-parameter AWQ ratio family.
    weight[:, ::7] *= 5.0
    inputs = torch.randn(samples, in_features, dtype=torch.float32)
    inputs[:, 1::5] *= 3.0
    reference = inputs @ weight.t()

    x_mean = inputs.abs().mean(0)
    grouped = weight.reshape(-1, group_size)
    w_mean = (
        grouped.abs() / (grouped.abs().amax(1, keepdim=True) + 1e-6)
    ).reshape_as(weight).mean(0)

    def scales_for_ratio(ratio: float) -> torch.Tensor:
        scales = (
            x_mean.pow(ratio) / (w_mean.pow(1.0 - ratio) + 1e-4)
        ).clamp(min=1e-4)
        return scales / (scales.max() * scales.min()).sqrt()

    def exact_loss(scales: torch.Tensor) -> float:
        scaled_weight = weight * scales.view(1, -1)
        effective_weight = _fake_quantize_asymmetric_groupwise(
            scaled_weight, group_size
        ) / scales.view(1, -1)
        return float((inputs @ effective_weight.t() - reference).pow(2).mean())

    grid = [(exact_loss(scales_for_ratio(i / 20)), i / 20) for i in range(20)]
    initial_loss, best_ratio = min(grid)
    initial_scales = scales_for_ratio(best_ratio)

    def differentiable_objective(log_correction: torch.Tensor) -> torch.Tensor:
        scales = channel_scales_from_log_correction(
            initial_scales, log_correction
        )
        scaled_weight = weight * scales.view(1, -1)
        effective_weight = _fake_quantize_asymmetric_groupwise(
            scaled_weight, group_size, ste=True
        ) / scales.view(1, -1)
        return (inputs @ effective_weight.t() - reference).pow(2).mean()

    refinement = refine_channel_scales(
        initial_scales,
        initial_loss,
        differentiable_objective,
        exact_loss,
        maxiter=maxiter,
        relative_bound=1.5,
        max_scale_change=max_scale_change,
    )
    print("Synthetic AWQ SciPy refinement")
    print(f"grid best ratio     : {best_ratio:.4f}")
    print(f"grid loss           : {initial_loss:.8f}")
    print(f"refined exact loss  : {refinement.candidate_loss:.8f}")
    print(f"relative improvement: {refinement.relative_improvement * 100:.4f}%")
    print(f"accepted            : {refinement.accepted}")
    print(f"rollback callback   : {refinement.callback_triggered}")
    if refinement.callback_reason:
        print(f"callback reason     : {refinement.callback_reason}")
    print(
        "SciPy iterations/evals:",
        refinement.iterations,
        "/",
        refinement.function_evaluations,
    )
    return refinement


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ SciPy scale-refinement demo")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maxiter", type=int, default=10_000_000)
    parser.add_argument("--max-scale-change", type=float, default=0.10)
    args = parser.parse_args()
    run_demo(
        seed=args.seed,
        maxiter=args.maxiter,
        max_scale_change=args.max_scale_change,
    )


if __name__ == "__main__":
    main()
