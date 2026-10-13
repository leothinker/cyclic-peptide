from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project src/ importable when run as a script.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hiro_ocsr import HiroOCSR  # type: ignore[import-not-found]


IMG = Path("data/processed/cells/PCTCN2025075389-ftappb-I100297_1036_struct.png")


def main() -> int:
    img_path = ROOT / IMG
    if not img_path.exists():
        print(f"[!] image missing: {IMG}")
        return 1

    # Allow local-only weights via HIRO_MODEL_DIR so an approved token
    # doesn't need network access on every run.
    model_path = os.environ.get("HIRO_MODEL_DIR", "PatSnap/Hiro-OCSR")

    print(f"[hiro] loading model from {model_path!r} ...")
    try:
        model = HiroOCSR(model_path=model_path, load=True)
    except Exception as exc:  # gated / 401 / network errors land here
        print("[hiro] cannot load weights:")
        print(f"        {type(exc).__name__}: {exc}")
        print()
        print("    PatSnap/Hiro-OCSR is a gated repo on Hugging Face.")
        print("    Once your access request is approved (see")
        print("    https://huggingface.co/settings/gated-repos), log in with:")
        print()
        print("        .venv\\Scripts\\python.exe -c \"from huggingface_hub import login; login(token='hf_xxx')\"")
        print()
        print("    Then re-run this script. The weights will cache to ~/.cache/huggingface.")
        return 0  # graceful exit; this is "expected" pre-approval

    print(f"[hiro] running inference on {IMG} ...")
    result = model.run({"image": str(img_path), "depict": False})

    if not result:
        print("[!] no result returned")
        return 1

    entry = result[0]
    if isinstance(entry, dict):
        smi = entry.get("SMILES")
        if not smi:
            print(f"[!] result has no SMILES field: {entry}")
            return 1
        print(f"[hiro] predicted SMILES ({len(smi)} chars): {smi}")
        return 0

    print(f"[!] unexpected result shape: {type(entry).__name__}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
