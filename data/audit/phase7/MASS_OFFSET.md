# Why the mass gate rejects every candidate

**Date:** 2026-11-04
**Script:** `experiments/phase7_mass_offset.py`
**Data:** `data/audit/phase7/mass_offset.csv`, `mass_offset_summary.txt`

## The question

The mass gate (`cpd.ms_validation`) compares the monoisotopic mass of
a candidate SMILES with the neutral mass implied by the patent's own
printed `m/z`.  It rejects every one of the 406 compounds.  That
number is either

* a statement about the *reader* -- the printed SMILES are wrong; or
* a statement about the *reference* -- the patent's `m/z` values were
  computed differently from the way the gate computes a mass.

The two have to be told apart before any of the gate's verdicts can be
used as evidence.

## What was measured

For every compound whose printed SMILES parses (171 of 406) and that
has an `m/z`:

| mass convention | median offset | median offset | within +-0.05 Da |
|---|---:|---:|---:|
| monoisotopic `[M+H]+` (what the gate uses) | **+0.1631 Da** | **+104.3 ppm** | **0 / 171** |
| rounded table `[M+H]+` (H = 1.008, S = 32.06) | +0.0082 Da | +5.1 ppm | 63 / 171 |
| average molecular weight `[M+H]+` | -0.9053 Da | -565.0 ppm | 0 / 171 |

The closest convention per compound is the rounded table for 153 of
the 171, the monoisotopic table for 15, and a doubly charged ion for
3.

## Why this is not a reading error

**The offset is far too small to be a character error.**  The lightest
atom is hydrogen at 1.00783 Da, and every other element is heavier
still, so a single wrong character in a printed SMILES changes the
mass by at least 1 Da.  The offset here is 0.13-0.20 Da.

**The offset is too tight to be noise.**  The middle half of the
offsets lie between +0.133 and +0.202 Da.  A random reading error
would scatter over whole daltons.

**The offset points at the mass table.**  Recomputing the same
molecules with H = 1.008 and S = 32.06 -- the rounded atomic weights a
molecular-weight calculator prints, rather than the monoisotopic ones
RDKit uses -- moves the median offset from +0.163 Da to +0.008 Da.

## What is still open

* The rounded table does not explain every compound: the residual
  spread is about +-0.07 Da, and 63 of 171 land inside the +-0.05 Da
  tolerance.  The table is therefore *consistent with* the patent's
  numbers, not proven to be the table the patent used.
* Three compounds have an `m/z` that is about half of the
  monoisotopic `[M+H]+` of their printed SMILES; they look like
  doubly charged ions.
* A least-squares fit of the offset against the element counts does
  not identify the table, because the compounds of this family have
  strongly correlated formulas (all of them have 70-108 carbons and
  12-13 nitrogens).  The fit is reported but not interpreted.

## What was *not* done

The tolerance was **not** widened and the gate was **not** changed.
Loosening the tolerance would hide the discrepancy instead of
explaining it, and a different mass table has to be a decision about
the chemistry, not a side effect of a pipeline change.

The practical consequence for now: **the `mass_mismatch` verdict is
not evidence that the printed SMILES are wrong.**  It says that the
candidate does not reproduce the patent's `m/z` under the
monoisotopic convention.  The printed SMILES remain a *text-level*
result, and the structure-level check -- comparing the graph with the
printed drawing -- is still not automated.

## The next step, if it is wanted

Make the mass convention an explicit, per-profile setting, with the
rounded table as a measured alternative and the current monoisotopic
table as the default, and report both verdicts.  That is a
chemistry decision and should be made explicitly rather than by
changing a number until the numbers look better.
