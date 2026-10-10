"""Optional DECIMER (OCSR) wrapper for 2D structure -> SMILES.

DECIMER-AI is a heavy install (PyTorch + ~2 GB of model weights). This
module degrades gracefully when the package is not present so the rest of
the pipeline keeps running -- ``smiles_decimer`` stays empty until DECIMER
is installed and the pipeline re-run.

The wrapper exposes both the bare SMILES string and the per-token
confidence that DECIMER-AI's ``predict_SMILES(confidence=True)`` returns.
Low-confidence tokens are the signature of phantom fragments (e.g. the
``CC.CC`` tail DECIMER appends to peptides that confuse its RDKit
canonicalisation downstream), so the caller can either log them or
truncate the SMILES at the first low-confidence segment break.

Install with one of::

    uv pip install --python .venv/Scripts/python.exe DECIMER
    .venv/Scripts/python.exe -m pip install DECIMER
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class DecimerPrediction:
    """One DECIMER result: the SMILES string and per-token confidences.

    ``tokens`` is a list of ``(token_str, conf)`` in DECIMER emission
    order, WITHOUT the ``<start>`` / ``<end>`` markers. ``conf`` is the
    softmax probability the transformer decoder assigned to that token.
    """

    smiles: str
    tokens: tuple[tuple[str, float], ...]

    @property
    def min_conf(self) -> float:
        """Lowest per-token confidence. ``0.0`` when tokens is empty."""
        return min((c for _, c in self.tokens), default=0.0)

    @property
    def mean_conf(self) -> float:
        """Mean per-token confidence, or 0.0 when tokens is empty."""
        if not self.tokens:
            return 0.0
        return sum(c for _, c in self.tokens) / len(self.tokens)


def _parse_prediction(raw) -> DecimerPrediction | None:
    """Coerce whatever DECIMER returns into a ``DecimerPrediction``.

    ``predict_SMILES`` returns a plain SMILES string when
    ``confidence=False`` and a ``(smiles, [(tok, conf), ...])`` tuple
    when ``confidence=True``. Normalise both into our richer type.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        smi = raw.strip()
        return DecimerPrediction(smiles=smi, tokens=()) if smi else None
    if isinstance(raw, tuple) and len(raw) == 2:
        smi_raw, tokens_raw = raw
        smi = (smi_raw or "").strip()
        if not smi:
            return None
        tokens: list[tuple[str, float]] = []
        for item in tokens_raw or ():
            try:
                tok, conf = item
            except (TypeError, ValueError):
                continue
            tokens.append((str(tok), float(conf)))
        return DecimerPrediction(smiles=smi, tokens=tuple(tokens))
    # Unknown shape -- be lenient.
    smi = str(raw).strip()
    return DecimerPrediction(smiles=smi, tokens=()) if smi else None


def predict(image_path: Path) -> DecimerPrediction | None:
    """Return the full DECIMER prediction for one structure PNG.

    Returns None when DECIMER is not installed, the image cannot be read,
    or the model produces no SMILES. Never raises -- callers can treat
    None as "not available, fall back to OCR".
    """
    engine = _ensure_engine()
    if engine is None:
        return None
    try:
        raw = engine(str(image_path), confidence=True)
    except TypeError:
        # Older DECIMER builds without the confidence kwarg -- fall back
        # to the string-only call so the wrapper still works.
        try:
            raw = engine(str(image_path))
        except Exception:  # pragma: no cover - model errors are not actionable
            return None
    except Exception:  # pragma: no cover - model errors are not actionable
        return None
    return _parse_prediction(raw)


def predict_smiles(image_path: Path) -> str | None:
    """Return just the DECIMER-predicted SMILES string, or None.

    Thin wrapper around :func:`predict` for callers that only need the
    string. New code should prefer :func:`predict` so per-token
    confidences are available for filtering and diagnostics.
    """
    pred = predict(image_path)
    return pred.smiles if pred is not None else None


def low_confidence_tokens(
    prediction: DecimerPrediction,
    threshold: float = 0.5,
) -> list[int]:
    """Indices in ``prediction.tokens`` whose confidence is below the threshold.

    Useful for locating the spot where DECIMER started hallucinating
    phantom fragments. ``threshold=0.5`` catches the common
    ``CC.CC``-tail failure mode without false-flagging genuinely
    unusual but real peptide bonds.
    """
    return [i for i, (_, c) in enumerate(prediction.tokens) if c < threshold]
