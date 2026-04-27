"""Magnetometer ellipsoid (hard + soft iron) calibration.

The raw magnetometer measures ``m = A B + b + noise`` where ``A`` is the
soft-iron transform (3×3, near-diagonal) and ``b`` is the hard-iron offset
(3,).  Calibrated samples ``m' = A_inv @ (m - b)`` should lie on a sphere
of constant radius, which the orientation filter expects.

Fit strategy
------------
A direct general-quadric least-squares fit is tempting (single closed-form
solve) but does *not* constrain the resulting Q to be positive definite.
On noisy real-world mag streams the unconstrained Q comes back indefinite
in practice, so we use a robust two-stage decomposition instead:

1. Coope's sphere fit gives a closed-form hard-iron centre.  The same
   linear system is well-conditioned even when the points lie on a
   distorted ellipsoid — the centre estimate matches the ellipsoid centroid
   to first order.
2. After translating samples to the estimated centre, the sample
   covariance is proportional to the ellipsoid's metric tensor.  An
   eigendecomposition gives the principal axes; rescaling by the inverse
   square-root of the eigenvalues maps the ellipsoid to a unit sphere.
   This gives a symmetric soft-iron correction A_inv = U diag(λ⁻½) Uᵀ.

Static six-face calibration only yields ~6 widely-spread samples (and ours
has none at all), but a full cycling section provides thousands of mag
readings at many heading/tilt combinations — more than enough for a stable
fit at this complexity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

_MIN_SAMPLES = 100
_MIN_SPAN = 5.0  # minimum range (max-min) on each axis to avoid degenerate fits
_DEGENERATE_COND = 1e8  # condition number above which we reject the fit


@dataclass
class MagCalibration:
    """Hard- and soft-iron correction for a magnetometer.

    Apply with :meth:`apply` to get unit-sphere mag samples regardless of
    rigid-body orientation.  ``residual`` is the post-fit standard deviation
    of corrected mag norms (≈0 means a perfect ellipsoid fit).
    """

    offset: np.ndarray  # (3,) hard-iron bias
    transform: np.ndarray  # (3, 3) soft-iron correction (A_inv)
    radius: float  # mean post-correction magnitude (uncalibrated units)
    residual: float  # std of post-correction magnitudes / radius
    n_samples: int

    @classmethod
    def identity(cls) -> "MagCalibration":
        return cls(
            offset=np.zeros(3),
            transform=np.eye(3),
            radius=0.0,
            residual=float("nan"),
            n_samples=0,
        )

    def apply(self, mag: np.ndarray) -> np.ndarray:
        """Return calibrated samples (same shape as input).  NaN rows pass through."""
        m = np.asarray(mag, dtype=float)
        out = np.full_like(m, np.nan)
        finite = np.all(np.isfinite(m), axis=1)
        if np.any(finite):
            out[finite] = (m[finite] - self.offset) @ self.transform.T
        return out

    def to_dict(self) -> dict:
        return {
            "offset": self.offset.tolist(),
            "transform": self.transform.tolist(),
            "radius": float(self.radius),
            "residual": float(self.residual),
            "n_samples": int(self.n_samples),
        }


def _coope_sphere_fit(pts: np.ndarray) -> np.ndarray:
    """Closed-form sphere centre via Coope (1993): linear lstsq on m^T m."""
    A = np.hstack([2.0 * pts, np.ones((pts.shape[0], 1))])
    b = np.sum(pts ** 2, axis=1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3]


def fit_mag_ellipsoid(mag: np.ndarray) -> MagCalibration:
    """Fit hard + soft iron correction from raw magnetometer samples.

    Returns an identity calibration when the sample set is too small or
    insufficiently diverse for a stable fit (logs the reason at WARNING).
    """
    m = np.asarray(mag, dtype=float)
    finite = np.all(np.isfinite(m), axis=1)
    pts = m[finite]
    if pts.shape[0] < _MIN_SAMPLES:
        log.warning("Mag ellipsoid fit skipped: only %d finite samples (< %d).",
                    pts.shape[0], _MIN_SAMPLES)
        return MagCalibration.identity()

    span = pts.max(axis=0) - pts.min(axis=0)
    if np.any(span < _MIN_SPAN):
        log.warning(
            "Mag ellipsoid fit skipped: per-axis span %s below threshold %.1f.",
            np.round(span, 2).tolist(), _MIN_SPAN,
        )
        return MagCalibration.identity()

    # 1. Hard-iron (sphere centre) via the linear, well-conditioned Coope fit.
    offset = _coope_sphere_fit(pts)
    centred = pts - offset

    # 2. Soft-iron via least-squares fit of the symmetric metric tensor M:
    #    centred^T M centred = 1 for samples on the calibrated unit sphere.
    # Six unknowns (M is 3×3 symmetric) → linear lstsq, no positivity
    # constraint built in but normally produces positive-definite M for
    # well-distributed mag samples.  This is more accurate than scaling by
    # the covariance because it minimises the actual ellipsoid residual
    # rather than assuming uniform surface sampling.
    x, y, z = centred[:, 0], centred[:, 1], centred[:, 2]
    D_mat = np.column_stack([
        x * x, y * y, z * z,
        2.0 * x * y, 2.0 * x * z, 2.0 * y * z,
    ])
    rhs = np.ones(D_mat.shape[0])
    try:
        params, *_ = np.linalg.lstsq(D_mat, rhs, rcond=None)
    except np.linalg.LinAlgError as exc:
        log.warning("Mag soft-iron lstsq failed: %s", exc)
        return MagCalibration.identity()
    M = np.array([
        [params[0], params[3], params[4]],
        [params[3], params[1], params[5]],
        [params[4], params[5], params[2]],
    ])

    eigvals, eigvecs = np.linalg.eigh(M)
    if np.any(eigvals <= 0) or not np.all(np.isfinite(eigvals)):
        log.warning("Mag soft-iron M non-positive-definite (eig=%s).",
                    np.round(eigvals, 8).tolist())
        return MagCalibration.identity()

    cond = float(eigvals.max() / max(eigvals.min(), 1e-30))
    if cond > _DEGENERATE_COND:
        log.warning("Mag soft-iron M degenerate (condition %.2e).", cond)
        return MagCalibration.identity()

    # A_inv = M^{1/2} via eigendecomposition; symmetric by construction.
    A_inv = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    corrected = centred @ A_inv.T
    norms = np.linalg.norm(corrected, axis=1)
    radius = float(np.mean(norms))
    residual = float(np.std(norms))

    log.info(
        "Mag ellipsoid fit: n=%d, |b|=%.2f, residual=%.4f (cond=%.1f).",
        pts.shape[0], float(np.linalg.norm(offset)), residual, cond,
    )

    return MagCalibration(
        offset=offset,
        transform=A_inv,
        radius=radius,
        residual=residual,
        n_samples=int(pts.shape[0]),
    )
