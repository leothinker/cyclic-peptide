"""Chemical structure OCR (image -> SMILES)."""
from cpd.ocr.base import StructureReader
from cpd.ocr.decimer import DecimerReader

__all__ = ["DecimerReader", "StructureReader"]
