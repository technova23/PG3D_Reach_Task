from __future__ import annotations

import pytest

from pg3d.eval import calibrate_verifier_buffer


def test_verifier_buffer_adds_padding_and_rounds_up() -> None:
    result = calibrate_verifier_buffer([0.001, 0.009, 0.011])

    assert result.optimistic_error_p95_m == pytest.approx(0.0108)
    assert result.verification_buffer_m == pytest.approx(0.02)
    assert result.verifier_valid is True


def test_verifier_calibration_stops_at_three_centimeters() -> None:
    result = calibrate_verifier_buffer([0.03, 0.03])

    assert result.verifier_valid is False
    assert result.verification_buffer_m == pytest.approx(0.035)
