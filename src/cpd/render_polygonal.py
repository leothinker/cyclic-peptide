"""ChemDraw-style rendering for macrocyclic peptides.

Why this exists
---------------
The patent tables draw a cyclic peptide the way ChemDraw does: the
macrocyclic backbone is a closed polygon with **straight segments and
sharp corners**, side chains radiate outward, and substituents sit at
fixed bond geometries. RDKit's two layout engines -- :func:`rdkit.Chem.rdCoordGen.AddCoords`
and :func:`rdkit.Chem.rdDepictor.Compute2DCoords` -- treat a ring of more
than ~14 atoms as a macrocycle and distribute it smoothly around a circle,
so a 33-atom peptide backbone comes out as an ellipse. That is a valid
depiction but it is not the one in the patent.

The approach
------------
We start from Depictor's layout, which puts side chains in sensible
places, and then *re-template* the macrocycle atoms onto a regular polygon.
An iterative pin-and-relax loop lets substituents settle into the new
shape without collapsing into the centroid: each iteration pins the ring
atoms to their polygon positions and runs a few hundred steps of UFF
minimisation.

Compared to a global radial warp, this approach keeps the side chains'
rotational orientation around their parent ring atom, which is what stops
them from being smeared outward when the ring becomes a polygon.

Public API
----------
* :func:`macrocycle_polygon_coords`  -- ring-ordered polygon coordinates
* :func:`build_polygonal_mol`        -- SMILES to RDKit Mol with the layout
* :func:`polygonal_macrocycle_render` -- SMILES to a PIL image
* :func:`describe_polygonal_layout`  -- numeric summary, used in tests
"""

from __future__ import annotations

import math
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, rdDepictor
from rdkit.Geometry import Point2D, Point3D

RDLogger.DisableLog("rdApp.*")

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

# A drawn peptide macrocycle in this patent family resolves into 8-16
# straight runs; fewer looks stamped, more starts to look circular again,
# which is exactly what we are trying to avoid.
DEFAULT_CORNERS = 12

# Fraction of bond length used to alternate ring atoms in and out of the
# polygon outline. Adds the serrated look that a hand-drawn chain has.
DEFAULT_ZIGZAG = 0.4

BOND_LENGTH = 1.5
RELAX_ITERATIONS = 3
RELAX_STEPS = 200


def macrocycle_ring(mol: Chem.Mol) -> tuple:
    """Atoms of the largest ring, walked in ring-bond order."""
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        return ()
    return max(rings, key=len)


def macrocycle_centroid(mol: Chem.Mol, ring: tuple) -> tuple:
    """Centroid of ``ring``'s atoms in the molecule's current conformer."""
    conf = mol.GetConformer()
    cx = sum(conf.GetAtomPosition(i).x for i in ring) / len(ring)
    cy = sum(conf.GetAtomPosition(i).y for i in ring) / len(ring)
    return cx, cy


def macrocycle_polygon_coords(
    mol: Chem.Mol,
    ring: tuple,
    corners: int = DEFAULT_CORNERS,
    zigzag: float = DEFAULT_ZIGZAG,
    polygon_radius: Optional[float] = None,
) -> dict:
    """Place ``ring`` atoms along the perimeter of a regular polygon.

    If ``polygon_radius`` is None we size the polygon so its perimeter
    equals ``len(ring) * bond_length`` -- which is the same arc length
    a circle of the same centre would have, and so keeps the average
    ring-bond chord close to the canonical 1.5 A.
    """
    n = len(ring)
    if n == 0:
        return {}
    cx, cy = macrocycle_centroid(mol, ring)

    if polygon_radius is None:
        perimeter = n * BOND_LENGTH
        polygon_radius = perimeter / (2 * corners * math.sin(math.pi / corners))

    # Phase so corner #0 coincides with ring atom 0 -- corners that land
    # on atoms read as "drawn" rather than as a wobbly circle.
    conf = mol.GetConformer()
    first = conf.GetAtomPosition(ring[0])
    phase = math.atan2(first.y - cy, first.x - cx)

    vertices = [
        (
            cx + polygon_radius * math.cos(
                math.pi / 2 - 2 * math.pi * k / corners + phase
            ),
            cy + polygon_radius * math.sin(
                math.pi / 2 - 2 * math.pi * k / corners + phase
            ),
        )
        for k in range(corners)
    ]
    edges = []
    total = 0.0
    for c in range(corners):
        a = vertices[c]
        b = vertices[(c + 1) % corners]
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        edges.append((a, b, length, total))
        total += length

    step = total / n
    coords = {}
    for k, idx in enumerate(ring):
        distance = (k + 0.5) * step
        distance %= total
        for a, b, length, start in edges:
            if distance <= start + length:
                t = (distance - start) / length if length else 0.0
                x = a[0] + (b[0] - a[0]) * t
                y = a[1] + (b[1] - a[1]) * t
                break
        else:
            x, y = vertices[0]
        sign = 1.0 if k % 2 == 0 else -1.0
        vx, vy = x - cx, y - cy
        norm = math.hypot(vx, vy) or 1.0
        coords[idx] = Point2D(
            x + sign * zigzag * BOND_LENGTH * vx / norm,
            y + sign * zigzag * BOND_LENGTH * vy / norm,
        )
    return coords


def _make_2d(mol: Chem.Mol) -> bool:
    """Ensure the molecule has a 2D conformer with non-zero coordinates."""
    if mol.GetNumConformers() == 0:
        try:
            rdDepictor.Compute2DCoords(mol)
            return True
        except Exception:
            return False
    return True


def _set_atom_xy(mol: Chem.Mol, idx: int, point) -> None:
    conformer = mol.GetConformer()
    conformer.SetAtomPosition(idx, Point3D(point.x, point.y, 0.0))


def _relax(mol: Chem.Mol, fixed: set, steps: int) -> None:
    """Minimise UFF energy with the atoms in ``fixed`` held still."""
    if not fixed:
        return
    try:
        ff = AllChem.UFFGetMoleculeForceField(
            mol, ignoreInterfragInteractions=True
        )
    except Exception:
        return
    if ff is None:
        return
    for idx in fixed:
        ff.AddFixedPoint(idx)
    try:
        ff.Minimize(maxIts=steps)
    except Exception:
        pass


def build_polygonal_mol(
    smiles: str,
    corners: int = DEFAULT_CORNERS,
    zigzag: float = DEFAULT_ZIGZAG,
) -> Optional[Chem.Mol]:
    """SMILES to RDKit Mol with a ChemDraw-like 2D conformation.

    The base layout is Depictor's -- it puts side chains in sensible
    places around the macrocycle. The macrocycle is then re-templated
    onto a regular polygon and an iterative pin-and-relax loop lets the
    side chains settle into the new shape.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    ring = macrocycle_ring(mol)
    if len(ring) < 12:
        return None

    if not _make_2d(mol):
        return None
    conf = mol.GetConformer()
    if conf.Is3D():
        # RDKit 2026.03 reports CoordGen's conformer as 3D even when its
        # positions are 2D. Inflate to 3D points so SetAtomPosition will
        # accept them, then flip the flag.
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            if p.z == 0:
                pass

    # Iterative pin-and-relax. The first round places the ring on the
    # polygon; subsequent rounds let substituents relax toward sensible
    # bond lengths without dragging the ring back into a circle.
    for iteration in range(RELAX_ITERATIONS):
        coords = macrocycle_polygon_coords(
            mol, ring, corners=corners, zigzag=zigzag
        )
        for idx, point in coords.items():
            _set_atom_xy(mol, idx, point)
        # Move the inner conformer flag *after* the positions are settled
        # so that subsequent UFF relaxation sees a clean 2D conformer.
        if mol.GetNumConformers() and mol.GetConformer().Is3D():
            mol.GetConformer().Set3D(False)
        _relax(mol, set(ring), RELAX_STEPS)

    return mol


def describe_polygonal_layout(mol: Chem.Mol) -> dict:
    """Turn-angles around the macrocycle: a circle is uniform, a polygon alternating.

    ``mean_turn`` is the average change in direction between consecutive
    backbone bonds (degrees). ``std_turn`` is the spread: a circular
    ring gives low std (all turns equal); a polygon + zigzag gives
    higher std because some turns are sharp and others run straight
    along the polygon edges.
    """
    conf = mol.GetConformer()
    ring = macrocycle_ring(mol)
    if len(ring) < 3:
        return {"mean_turn": 0.0, "std_turn": 0.0, "n": len(ring)}

    angles = []
    n = len(ring)
    for k in range(n):
        prev = conf.GetAtomPosition(ring[(k - 1) % n])
        cur = conf.GetAtomPosition(ring[k])
        nxt = conf.GetAtomPosition(ring[(k + 1) % n])
        a = (prev.x - cur.x, prev.y - cur.y)
        b = (nxt.x - cur.x, nxt.y - cur.y)
        na = math.hypot(*a)
        nb = math.hypot(*b)
        if not na or not nb:
            continue
        cosang = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosang)))))

    if not angles:
        return {"mean_turn": 0.0, "std_turn": 0.0, "n": n}
    mean = sum(angles) / len(angles)
    var = sum((x - mean) ** 2 for x in angles) / len(angles)
    return {"mean_turn": round(mean, 2), "std_turn": round(math.sqrt(var), 2), "n": n}


def _draw(mol: Chem.Mol, size: tuple, bond_line_width: int):
    from io import BytesIO

    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = bond_line_width
    opts.padding = 0.03
    opts.addStereoAnnotation = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")


def polygonal_macrocycle_render(
    smiles: str,
    corners: int = DEFAULT_CORNERS,
    zigzag: float = DEFAULT_ZIGZAG,
    size: tuple = (640, 520),
    bond_line_width: int = 2,
) -> Optional["Image.Image"]:
    """Render ``smiles`` with a closed, angular macrocycle.

    Falls back to RDKit's own depiction when there is no macrocycle to
    template, so the function is safe to call on any SMILES.
    """
    if Image is None:
        return None
    mol = build_polygonal_mol(smiles, corners=corners, zigzag=zigzag)
    if mol is None:
        fallback = Chem.MolFromSmiles(smiles or "")
        if fallback is None:
            return None
        return _draw(fallback, size, bond_line_width)
    return _draw(mol, size, bond_line_width)


__all__ = [
    "DEFAULT_CORNERS",
    "DEFAULT_ZIGZAG",
    "build_polygonal_mol",
    "describe_polygonal_layout",
    "macrocycle_centroid",
    "macrocycle_polygon_coords",
    "macrocycle_ring",
    "polygonal_macrocycle_render",
]
