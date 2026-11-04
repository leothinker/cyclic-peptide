"""Zoom one region of a captured WO2022251372 SMILES cell.

The characters of a printed SMILES are ambiguous at page scale, so a
transcription has to be checked against magnified pixels.  This script
cuts an arbitrary rectangle of a captured cell and writes it at 8x.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_patent2_zoom_region.py 10 1 0 460 0 200
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "audit" / "phase9" / "patent2_zoom"


def main() -> int:
    args = sys.argv[1:]
    compound = args[0]
    band = int(args[1])
    x, y, width, height = (int(value) for value in args[2:6])

    source = OUT / f"cmpd{compound}_band{band}.png"
    image = cv2.imread(str(source))
    if image is None:
        print(f"cannot read {source}")
        return 2
    crop = image[y:y + height, x:x + width]
    if crop.size == 0:
        print("region is empty")
        return 2
    big = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    target = OUT / f"cmpd{compound}_band{band}_zoom_{x}_{y}.png"
    cv2.imwrite(str(target), big)
    print(f"-> {target}  {big.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
