"""Write every printed line of a captured cell as its own PNG.

Reading a printed SMILES needs the lines separately: a mistake at the
end of one line is easy to miss when the whole cell is on one page.
This script splits a captured band into its ink bands and writes each
one at 6x, so the characters can be read one line at a time.

Usage::

    .venv\\Scripts\\python.exe experiments/phase9_patent2_zoom_rows.py 28 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.lines import ink_bands  # noqa: E402

OUT = REPO / "data" / "audit" / "phase9" / "patent2_zoom"


def main() -> int:
    compound = sys.argv[1]
    band = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    source = OUT / f"cmpd{compound}_band{band}.png"
    image = cv2.imread(str(source))
    if image is None:
        print(f"cannot read {source}")
        return 2

    # The saved PNG is already 3x; split it into ink bands at that
    # scale and write every band at 6x.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for index, (top, bottom) in enumerate(ink_bands(gray)):
        crop = image[top:bottom, :]
        big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        target = OUT / f"cmpd{compound}_band{band}_line{index:02d}.png"
        cv2.imwrite(str(target), big)
        print(f"-> {target}  {big.shape}  rows {top}..{bottom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
