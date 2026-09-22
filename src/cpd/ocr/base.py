"""Chemical structure OCR interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class StructureReader(ABC):
    """Image -> SMILES."""

    @abstractmethod
    def read(self, image_path: Path) -> str:
        """Return a SMILES string for a chemical structure figure."""
