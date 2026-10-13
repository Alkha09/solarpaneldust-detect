"""Synthetic Random Forest estimate for coverage-based efficiency loss."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


SIMULATION_FORMULA = "Estimated loss (%) = 0.2 x dust coverage (%)"
SIMULATION_SOURCE = (
    "Synthetic Random Forest estimate from the 0.2 x coverage assumption; "
    "not measured or calibrated on panel output data."
)

_coverage_samples = np.linspace(0.0, 100.0, 1001, dtype=np.float32).reshape(-1, 1)
_simulated_loss_samples = 0.2 * _coverage_samples[:, 0]
_estimator = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=1,
)
_estimator.fit(_coverage_samples, _simulated_loss_samples)


def estimate_efficiency_loss(dust_coverage_percent: float) -> float:
    """Estimate loss with an RF trained on the explicitly assumed synthetic rule."""
    coverage = float(dust_coverage_percent)
    if not np.isfinite(coverage) or not 0.0 <= coverage <= 100.0:
        raise ValueError("Dust coverage must be between 0 and 100 percent.")
    if coverage in (0.0, 100.0):
        return 0.2 * coverage
    prediction = float(_estimator.predict(np.array([[coverage]], dtype=np.float32))[0])
    return round(max(0.0, min(20.0, prediction)), 2)
