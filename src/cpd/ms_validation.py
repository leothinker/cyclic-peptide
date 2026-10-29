"""Validate a SMILES against the patent's reported LC-MS m/z.

Why this is the strongest check available
-----------------------------------------
The patent table gives an experimental ``m/z`` and an ionisation mode for
every compound.  That number was measured on a real instrument, so it is
ground truth that is completely independent of DECIMER, of OCR, and of every
repair heuristic we have written.  If a candidate SMILES reproduces the
reported mass to within a few ppm, the candidate is almost certainly the
right molecule.  If it is off by 100 Da, the candidate is wrong no matter
how plausible its ring system looks.

This is the gate that exposes how much of the macrocycle "repair" work was
self-confirming: several heuristics produced tidy macrocycles whose formula
never matched the measured mass.

Ion adducts
-----------
``[M+H]+``      neutral M = m/z - 1.00728
``[M-H]-``      neutral M = m/z + 1.00728
``[M+2H]2+``    neutral M = (m/z - 1.00728) * 2
``[M+3H]3+``    neutral M = (m/z - 1.00728) * 3
``[M+Na]+``     neutral M = m/z - 22.98922
``[M+K]+``      neutral M = m/z - 38.96316
``[M+NH4]+``    neutral M = m/z - 18.03382
``[M+CH3OH+H]+`` neutral M = m/z - 33.02657
``[M+HCOO]-``   neutral M = m/z + 44.99820

Public API
----------
* :func:`parse_polarity`    -- ion label -> charge and adduct delta
* :func:`neutral_mass_from_mz` -- observed m/z -> implied neutral monoisotopic mass
* :func:`mass_error_ppm`    -- measured vs expected error in ppm
* :func:`validate_against_ms` -- full verdict for one candidate SMILES
"""

from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

PROTON_MASS = 1.007276466
SODIUM_MASS = 22.98976928
POTASSIUM_MASS = 38.9637069
AMMONIUM_MASS = 18.033823
METHANOL_MASS = 32.026215
FORMATE_MASS = 44.998201

# Adduct name -> signed mass contributed per unit of charge.  Losses
# (a proton or a formate leaving the molecule) are negative, which lets one
# formula handle cations and anions alike.
_ADDUCTS: dict[str, float] = {
    "H": PROTON_MASS,
    "Na": SODIUM_MASS,
    "K": POTASSIUM_MASS,
    "NH4": AMMONIUM_MASS,
    "CH3OH+H": METHANOL_MASS + PROTON_MASS,
    "HCOO": -FORMATE_MASS,
    "FA-H": -FORMATE_MASS,
}


def parse_polarity(label: str) -> Optional[tuple[float, int]]:
    """Return ``(signed_adduct_delta, signed_charge)`` for an ion label.

    ``signed_adduct_delta`` is the mass added to the neutral molecule per
    unit of charge (negative when the ion is a *loss* such as ``[M-H]-``),
    and ``signed_charge`` is negative for anions.  The neutral mass then
    follows from :func:`neutral_mass_from_mz`.  Returns ``None`` for an
    unrecognised label so callers can count it as "not checkable" rather
    than as a failure.
    """
    if not label:
        return None
    text = label.replace(" ", "")
    # Charge multiplier and sign, e.g. the "3+" of "[M+3H]3+".
    m = re.search(r"\]\s*(\d*)([+-])\s*$", text)
    multiplier = 1
    charge_sign = 1
    if m:
        multiplier = int(m.group(1)) if m.group(1) else 1
        charge_sign = 1 if m.group(2) == "+" else -1
    charge = multiplier * charge_sign

    # Adduct body inside the brackets, e.g. "M+2H", "M-H", "M+Na".
    body = re.sub(r"\[\s*(.*?)\s*\][^]]*$", lambda g: g.group(1), text)
    body = body.strip()
    if body in ("M", ""):
        # A bare "[M+]" carries no adduct information; assume protonation.
        return (PROTON_MASS, charge)

    # Longest key first: "HCOO" must win over the "H" it contains, otherwise
    # a formate adduct would be parsed as a plain protonation.
    for key in sorted(_ADDUCTS, key=len, reverse=True):
        if key in body:
            # The sign of the charge already encodes gain vs loss, so the
            # adduct mass keeps its own sign: a proton is +1.00728 whether
            # it is added ([M+H]+) or lost ([M-H]-), and a formate *anion*
            # is genuinely -44.998.
            return (_ADDUCTS[key], charge)
    return None


def neutral_mass_from_mz(mz: float, polarity: str) -> Optional[float]:
    """Return the neutral monoisotopic mass implied by an observed ``m/z``.

    The ion obeys ``m/z = (M + z * delta) / |z|``, so
    ``M = |z| * m/z - z * delta``.  That single expression covers
    ``[M+H]+``, ``[M+3H]3+``, ``[M-H]-`` and ``[M+Na]+`` alike.
    """
    parsed = parse_polarity(polarity)
    if parsed is None or mz is None:
        return None
    delta, charge = parsed
    if charge == 0:
        return None
    return abs(charge) * mz - charge * delta


def mass_error_ppm(expected: float, observed: float) -> Optional[float]:
    """Signed mass error in ppm; positive means ``observed`` is too heavy."""
    if not expected:
        return None
    return (observed - expected) / expected * 1e6


def mz_printing_tolerance_ppm(mz: float) -> float:
    """Return the ppm tolerance implied by how many decimals ``mz`` has.

    A patent cell that prints ``1610.9`` carries an absolute uncertainty of
    about +/- 0.05 Da, which at 1610 Da is roughly 31 ppm.  A cell that
    prints ``1610.93`` is good to +/- 0.005 Da, or 3 ppm.  Comparing a
    computed mass against a printed one therefore needs a tolerance that
    follows the printed precision, otherwise the check rejects correct
    structures purely for the formatting of the table.

    Returns 5000 ppm when the value has no decimal point at all, which is
    the worst case a printed table can realistically produce.
    """
    if not mz or mz <= 0:
        return 5000.0
    text = repr(float(mz))
    if "." not in text:
        return 5000.0
    decimals = len(text.split(".")[1].rstrip("0")) or 1
    half_unit = 0.5 * (10.0**-decimals)
    return half_unit / mz * 1e6


def monoisotopic_mass(smiles: str) -> Optional[float]:
    """Exact monoisotopic mass of a SMILES, or ``None`` if unparseable."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return rdMolDescriptors.CalcExactMolWt(mol)
    except Exception:
        return None


def validate_against_ms(
    smiles: str,
    mz: Optional[float],
    polarity: str,
    tolerance_ppm: float | None = None,
) -> dict:
    """Check a candidate SMILES against the patent's reported LC-MS mass.

    Returns a dict with ``verdict`` in ``{"match", "mismatch", "unknown"}``,
    the expected/observed masses, the ppm error, and the implied formula
    when the molecule parses.

    ``tolerance_ppm`` defaults to :func:`mz_printing_tolerance_ppm`, i.e.
    the uncertainty implied by how many decimals the patent printed.  Pass
    an explicit value to tighten or loosen the gate.

    """
    result: dict = {
        "verdict": "unknown",
        "expected_mass": None,
        "observed_mass": None,
        "error_ppm": None,
        "formula": None,
        "polarity": polarity,
        "mz": mz,
    }
    obs = monoisotopic_mass(smiles)
    if obs is None:
        return result
    result["observed_mass"] = round(obs, 4)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        result["formula"] = rdMolDescriptors.CalcMolFormula(mol)
    expected = neutral_mass_from_mz(mz, polarity) if mz else None
    if expected is None:
        return result
    result["expected_mass"] = round(expected, 4)
    err = mass_error_ppm(obs, expected)
    if err is None:
        return result
    result["error_ppm"] = round(err, 1)
    tol = (
        tolerance_ppm
        if tolerance_ppm is not None
        else mz_printing_tolerance_ppm(mz)
    )
    result["tolerance_ppm"] = round(tol, 1)
    result["verdict"] = "match" if abs(err) <= tol else "mismatch"
    return result


__all__ = [
    "mz_printing_tolerance_ppm",
    "monoisotopic_mass",
    "mass_error_ppm",
    "neutral_mass_from_mz",
    "parse_polarity",
    "validate_against_ms",
]
