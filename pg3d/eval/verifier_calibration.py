from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VerifierCalibration:
    sample_count: int
    optimistic_error_p95_m: float
    verification_buffer_m: float
    verifier_valid: bool
    stop_threshold_m: float = 0.03

    def to_json(self) -> dict[str, int | float | bool]:
        return {
            "sample_count": int(self.sample_count),
            "optimistic_error_p95_m": float(self.optimistic_error_p95_m),
            "verification_buffer_m": float(self.verification_buffer_m),
            "authoritative_executed_clearance_m": 0.03,
            "verifier_valid": bool(self.verifier_valid),
            "stop_threshold_m": float(self.stop_threshold_m),
        }


def calibrate_verifier_buffer(
    optimistic_errors_m: Iterable[float],
    *,
    quantile: float = 0.95,
    padding_m: float = 0.005,
    rounding_m: float = 0.005,
    stop_threshold_m: float = 0.03,
) -> VerifierCalibration:
    errors = np.asarray(list(optimistic_errors_m), dtype=np.float64)
    if not len(errors):
        raise ValueError("verifier calibration requires optimistic-error samples")
    if not np.isfinite(errors).all() or np.any(errors < 0.0):
        raise ValueError("optimistic errors must be finite and non-negative")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    if padding_m < 0.0 or rounding_m <= 0.0 or stop_threshold_m <= 0.0:
        raise ValueError("padding, rounding, and stop threshold must be valid")
    error_p95 = float(np.quantile(errors, quantile))
    buffer_m = math.ceil((error_p95 + padding_m) / rounding_m - 1e-12) * rounding_m
    return VerifierCalibration(
        sample_count=int(len(errors)),
        optimistic_error_p95_m=error_p95,
        verification_buffer_m=float(buffer_m),
        verifier_valid=error_p95 < stop_threshold_m,
        stop_threshold_m=stop_threshold_m,
    )
