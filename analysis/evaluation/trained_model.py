"""Save and load fully-trained classification pipelines after evaluation.

The CV loop in experiments.py evaluates generalisation via GroupKFold but
discards the fold pipelines afterwards.  This module trains a *final* pipeline
on the complete labelled dataset (same preprocessing, same hyperparameters) and
serialises it together with the metadata needed to apply it to new windows.

Typical usage
-------------
In run_evaluation (experiments.py):
    save_trained_model(X, y, le, feature_names, ...)

Loading for inference:
    tm = load_trained_model(path)
    labels = tm.predict(feature_df)
    probas = tm.predict_proba(feature_df)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common.paths import project_relative_path

log = logging.getLogger(__name__)


def _display_path(path: Path) -> str:
    try:
        return project_relative_path(path)
    except ValueError:
        return str(path)

_MODEL_SUFFIX = ".joblib"
_META_SUFFIX = "_meta.json"


@dataclass
class TrainedModel:
    """A fully-fitted classification pipeline with associated metadata."""

    pipeline: Pipeline
    label_col: str
    classes: list[str]
    feature_names: list[str]
    feature_set: str
    model_name: str
    trained_at_utc: str
    n_training_samples: int
    transition_matrix: list[list[float]] | None = None

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _align_features(self, df: pd.DataFrame) -> np.ndarray:
        """Select and order feature columns, filling missing ones with NaN."""
        out = np.full((len(df), len(self.feature_names)), np.nan)
        for i, col in enumerate(self.feature_names):
            if col in df.columns:
                out[:, i] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        return out

    def predict(self, df: pd.DataFrame) -> list[str]:
        """Return predicted class labels for each row in *df*."""
        X = self._align_features(df)
        encoded = self.pipeline.predict(X)
        le = LabelEncoder()
        le.classes_ = np.array(self.classes)
        return le.inverse_transform(encoded).tolist()

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predicted class probabilities as a DataFrame (one column per class)."""
        if not hasattr(self.pipeline.named_steps.get("clf", object()), "predict_proba"):
            raise NotImplementedError(
                f"Model {self.model_name!r} does not support predict_proba."
            )
        X = self._align_features(df)
        proba = self.pipeline.predict_proba(X)
        clf = self.pipeline.named_steps["clf"]
        col_order = list(clf.classes_)
        le = LabelEncoder()
        le.classes_ = np.array(self.classes)
        col_names = le.inverse_transform(col_order).tolist()
        return pd.DataFrame(proba, columns=col_names, index=df.index)


def estimate_transition_matrix(
    y: np.ndarray,
    section_ids: np.ndarray,
    n_classes: int,
    smoothing: float = 1.0,
) -> list[list[float]]:
    """Estimate a row-stochastic label transition matrix from training windows.

    Counts consecutive label pairs within each section (never across section
    boundaries) and applies Laplace smoothing so that no transition has zero
    probability.  *y* must contain integer-encoded labels in ``[0, n_classes)``.
    """
    counts = np.full((n_classes, n_classes), smoothing)
    for sid in np.unique(section_ids):
        labels = y[section_ids == sid]
        for a, b in zip(labels[:-1], labels[1:]):
            counts[a, b] += 1
    trans = counts / counts.sum(axis=1, keepdims=True)
    return trans.tolist()


def train_final_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    model: object,
) -> Pipeline:
    """Fit a fresh imputer + scaler + classifier on the full dataset."""
    from sklearn.base import clone
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clone(model)),
        ]
    )
    pipe.fit(X, y)
    return pipe


def save_trained_model(
    X: np.ndarray,
    y: np.ndarray,
    model: object,
    *,
    label_encoder: LabelEncoder,
    feature_names: list[str],
    label_col: str,
    feature_set: str,
    model_name: str,
    output_dir: Path,
    section_ids: np.ndarray | None = None,
) -> Path:
    """Train on *all* data and serialise the pipeline + metadata.

    Returns the path to the ``.joblib`` file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = train_final_pipeline(X, y, model)

    stem = f"trained_model_{feature_set}"
    model_path = output_dir / f"{stem}{_MODEL_SUFFIX}"
    meta_path = output_dir / f"{stem}{_META_SUFFIX}"

    joblib.dump(pipeline, model_path)
    log.info(
        "Saved trained pipeline (%s / %s) → %s",
        feature_set,
        model_name,
        _display_path(model_path),
    )

    transition_matrix: list[list[float]] | None = None
    if section_ids is not None:
        transition_matrix = estimate_transition_matrix(
            y, section_ids, n_classes=len(label_encoder.classes_)
        )

    meta = {
        "label_col": label_col,
        "classes": list(label_encoder.classes_),
        "feature_names": feature_names,
        "feature_set": feature_set,
        "model_name": model_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_training_samples": int(len(y)),
        "transition_matrix": transition_matrix,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.debug("Saved model metadata → %s", _display_path(meta_path))

    return model_path


def load_trained_model(model_path: Path | str) -> TrainedModel:
    """Load a saved pipeline and its metadata.

    *model_path* can point to either the ``.joblib`` file or its metadata
    ``.json`` sibling — the other file is resolved automatically.
    """
    model_path = Path(model_path)

    if model_path.suffix == ".json":
        meta_path = model_path
        joblib_path = model_path.with_name(
            model_path.name.replace(_META_SUFFIX, _MODEL_SUFFIX)
        )
    else:
        joblib_path = model_path.with_suffix(_MODEL_SUFFIX)
        stem = model_path.stem
        meta_path = model_path.with_name(f"{stem}{_META_SUFFIX}")

    if not joblib_path.exists():
        raise FileNotFoundError(f"Trained model file not found: {joblib_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Model metadata file not found: {meta_path}")

    pipeline: Pipeline = joblib.load(joblib_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    log.info(
        "Loaded trained model (%s / %s) from %s",
        meta.get("feature_set"),
        meta.get("model_name"),
        _display_path(joblib_path),
    )
    return TrainedModel(
        pipeline=pipeline,
        label_col=meta["label_col"],
        classes=meta["classes"],
        feature_names=meta["feature_names"],
        feature_set=meta["feature_set"],
        model_name=meta["model_name"],
        trained_at_utc=meta["trained_at_utc"],
        n_training_samples=meta["n_training_samples"],
        transition_matrix=meta.get("transition_matrix"),
    )


def list_trained_models(output_dir: Path | str) -> list[Path]:
    """Return paths to all ``.joblib`` files under *output_dir* (recursive)."""
    return sorted(Path(output_dir).rglob(f"trained_model_*{_MODEL_SUFFIX}"))
