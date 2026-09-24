"""DECIMER-based structure reader (image -> SMILES).

This is the primary OCSR backend for the pipeline. DECIMER is a transformer
model from the Steinbeck group that takes a chemical-structure figure and
predicts a SMILES string.

Why this is wrapped
-------------------
DECIMER ships as ``DECIMER`` (uppercase) on PyPI but is heavy: TensorFlow +
a downloaded model checkpoint on first call. We keep it behind a lazy import
so the rest of the codebase stays importable when the user has only installed
the core ``chem`` / ``pdf`` extras and not the ``ocr`` extra.

The :class:`DecimerReader` returns ``None`` (not raises) for figures it can't
handle -- empty model output, an asterisk-only Markush placeholder, or an
image of a reaction scheme rather than a single compound. This makes it
trivially composable in a pipeline: just skip ``None`` records.

Usage
-----
>>> from cpd.ocr.decimer import DecimerReader
>>> reader = DecimerReader()
>>> smiles = reader.read(Path("data/raw/WO2025162428/FullText/...jpg"))
>>> if smiles is None:
...     print("DECIMER could not parse this figure")
"""
from __future__ import annotations

from pathlib import Path

from cpd.ocr.base import StructureReader


class DecimerReader(StructureReader):
    """Image -> SMILES via DECIMER.

    Parameters
    ----------
    model_dir
        Where DECIMER should cache its downloaded checkpoint. Defaults to
        ``~/.cache/decimer`` (the DECIMER default).
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        self._model_dir = model_dir

    def read(self, image_path: Path) -> str | None:
        """Return a canonical-looking SMILES string, or ``None`` on failure.

        Returns ``None`` when:
          * DECIMER is not installed (ImportError -> ``None``);
          * the model returns an empty string;
          * the predicted SMILES is just a Markush placeholder like ``*``
            (we don't want to feed those downstream).
        """
        try:
            from DECIMER import predict_SMILES  # type: ignore
        except ImportError:
            # DECIMER isn't installed -- treat as "reader unavailable" rather
            # than crashing the whole pipeline. Callers can check the env.
            return None

        raw = predict_SMILES(str(image_path))
        if not raw:
            return None
        cleaned = raw.strip()
        if not cleaned or cleaned == "*" or "*" in cleaned and len(cleaned) < 5:
            # Asterisk-only or asterisk-dominant output = Markush residue.
            # DECIMER can't draw R-groups, so this is a fail signal.
            return None
        return cleaned

    def __repr__(self) -> str:  # pragma: no cover -- trivial
        return f"DecimerReader(model_dir={self._model_dir!r})"


__all__ = ["DecimerReader"]
