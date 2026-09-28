"""Patent-document parsers (table extractor + header aliasing)."""
from cpd.parsers.headers import HEADER_ALIASES, canonical_field, is_known_header
from cpd.parsers.table_extractor import (
    AssayRecord,
    Column,
    CompoundRecord,
    RapidOcrTableExtractor,
)

__all__ = [
    "HEADER_ALIASES",
    "canonical_field",
    "is_known_header",
    "AssayRecord",
    "Column",
    "CompoundRecord",
    "RapidOcrTableExtractor",
]
