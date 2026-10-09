"""Optional DECIMER (OCSR) wrapper for 2D structure -> SMILES.

DECIMER-AI is a heavy install (PyTorch + ~2 GB of model weights). This
module degrades gracefully when the package is not present so the rest of
the pipeline keeps running -- ``smiles_decimer`` stays empty until DECIMER
is installed and the pipeline re-run.

Install with one of::

    uv pip install --python .venv/Scripts/python.exe DECIMER
    .venv/Scripts/python.exe -m pip install DECIMER
"""

from __future__ import annotations

from pathlib import Path

_predict = None
_checked = False


def _ensure_engine():
    """Lazy-load DECIMER; cache the import result."""
    global _predict, _checked
    if _checked:
        return _predict
    _checked = True
    try:
        from DECIMER import predict_SMILES  # type: ignore[import-not-found]

        _predict = predict_SMILES
    except Exception:  # pragma: no cover - environment specific
        _predict = None
    return _predict


def is_available() -> bool:
    """True iff DECIMER is currently installed and importable."""
    return _ensure_engine() is not None


def predict_smiles(image_path: Path) -> str | None:
    """Return DECIMER-predicted SMILES for one structure PNG, or None.

    Returns None when DECIMER is not installed, the image cannot be read,
    or the model produces no SMILES. Never raises -- callers can treat
    None as "not available, fall back to OCR".
    """
    engine = _ensure_engine()
    if engine is None:
        return None
    try:
        result = engine(str(image_path))
    except Exception:  # pragma: no cover - model errors are not actionable
        return None
    if not result:
        return None
    smi = result.strip() if isinstance(result, str) else str(result).strip()
    return smi or None
