"""Local, provenance-tracked public UCI classification datasets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _manifest() -> dict:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("Dataset manifest missing; run python data/download_datasets.py")
    return json.loads(path.read_text(encoding="utf-8"))


def available_datasets() -> list[str]:
    """Return locally registered dataset names, in alphabetical order."""
    return sorted(_manifest())


def metadata(name: str) -> dict:
    """Return source, licensing evidence, checksums, and preprocessing notes."""
    manifest = _manifest()
    if name not in manifest:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(manifest)}")
    return manifest[name]


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load unscaled X (float64), y (int64); leave missing values as NaN.

    Identifier removal and integer label encoding are dataset parsing, not
    learned preprocessing. Imputation, scaling and feature selection belong
    inside the training fold and are deliberately absent here.
    """
    info = metadata(name)
    path = DATA_DIR / "processed" / f"{name}.npz"
    with np.load(path, allow_pickle=False) as payload:
        X = np.asarray(payload["X"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.int64)
    if X.shape != (info["n_samples"], info["n_features"]) or y.shape != (len(X),):
        raise ValueError(f"Dataset shape disagrees with manifest: {name}")
    return X, y
