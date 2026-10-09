#!/usr/bin/env python3
"""
mesopore_carve.py

Carve a cylindrical mesopore through the centre of a zeolite (or other
Si/Al-O tetrahedral-framework) supercell, and cap the dangling bonds
created at the new pore surface with silanol (T-OH) groups.

Supports DL_POLY CONFIG files and GROMACS .gro files as input AND output,
auto-detected from file content/extension. The pore axis can be ANY
direction vector, not just a cell edge.

-------------------------------------------------------------------------
ALGORITHM
-------------------------------------------------------------------------
1. Parse atoms + cell from the input file.
2. Guess each atom's element from its name/label.
3. Build the T-O bond network (T = framework tetrahedral cations, default
   Si and Al) using a distance cutoff, correctly handling periodic images.
4. HYDROXYLATION-AWARE CARVING: decide which atoms actually get removed.
   The pore is NOT forced to be a perfect cylinder - atoms within
   +/- `--wiggle` Angstrom of the nominal pore wall may be rescued (kept
   instead of removed) if removing them would leave a surviving T atom
   with more than `--max-caps` capping (-OH) groups. See "HYDROXYLATION-
   AWARE CARVING" below for the details.
5. CONSISTENCY CLEANUP: any bridging O that would be removed but whose
   original T neighbours ALL survive is automatically restored (kept),
   regardless of the wiggle limit. See "SHARED-OXYGEN CLEANUP" below.
6. Using the ORIGINAL (pre-cut) bond list:
     a. Any surviving bridging O that lost one of its T neighbours becomes
        a silanol oxygen -> add one H, pointing along the REAL original
        bond direction from that O towards where its missing T neighbour
        used to sit (i.e. reusing the true local geometry, not a guess).
     b. Any surviving T atom that lost an O neighbour ENTIRELY (i.e. that
        O was itself removed) gets a brand new -OH group placed along the
        REAL original T-O bond direction for that specific missing
        oxygen - again reusing the true local geometry, so the new bond
        is (by construction) at the correct tetrahedral angle relative
        to whichever of the T atom's other bonds survived.
7. Write the result back out in CONFIG or .gro format (chosen by output
   file extension).

-------------------------------------------------------------------------
HYDROXYLATION-AWARE CARVING (step 4 in detail)
-------------------------------------------------------------------------
For a surviving T atom, each of its original O neighbours ends up in
exactly one of these states:
    - fully intact bridging O (both T-neighbours survive)         -> 0 caps
    - O removed entirely                                          -> 1 cap
    - O survives, but ITS OTHER T-neighbour was removed            -> 1 cap
(either way the T atom ends up with a terminal -OH substituent there)

`decide_keep_mask()` starts from the plain cylinder cut (radius = diameter/2)
and then, for every surviving T atom whose cap count exceeds `--max-caps`,
looks for the smallest possible "rescue" - the atom(s) that would need to
be kept instead of removed to eliminate one of the excess caps - and
performs that rescue ONLY if every atom involved lies within the wiggle
band (|radial distance - radius| <= wiggle), i.e. only if doing so keeps
the pore close to the requested cylinder. Atoms are only ever moved from
"remove" to "keep" (never the other way), so the pore can only get
locally smaller than requested, never bigger. If a T atom still exceeds
`--max-caps` after exhausting rescues within the wiggle band, it is
reported as a warning rather than silently left over-hydroxylated.

-------------------------------------------------------------------------
SHARED-OXYGEN CLEANUP (step 5 in detail)
-------------------------------------------------------------------------
A bridging O connects exactly two T atoms. If the O itself is removed but
BOTH of those T atoms survive, that is not a meaningful "cut" - the O just
happened to dip fractionally closer to the pore axis than either of its
two neighbours. Leaving it removed would force each of the two surviving
T atoms to independently generate a brand-new capping -OH aimed at the
same now-vacant point in space, i.e. two new atoms placed on top of each
other. `restore_fully_bridged_oxygens()` detects this and simply restores
(keeps) the O - no new atoms are needed, and no capping group is added to
either T atom, since the bridge is fully intact again. This check is
unconditional (not limited by --wiggle) because it is a correctness fix,
not a pore-shape trade-off.

-------------------------------------------------------------------------
IMPORTANT CAVEATS
-------------------------------------------------------------------------
* New atoms are given generic element-based names ("O_s", "H_s") with no
  force-field atom types/charges - you must assign these yourself
  afterwards (e.g. with a topology tool for your chosen FF).
* New T-OH bond LENGTHS are standardised to --to-bond / --oh-bond (not the
  real original T-O distance, which can vary slightly with local strain),
  but the bond DIRECTIONS are the real, original ones - so angles are
  physically correct even though lengths are idealised.
* Still not energy-minimised. Always run a short geometry optimisation /
  relaxation on the pore surface before production MD.
* Bonding is purely distance-based (T-O cutoff). Check --cutoff matches
  your framework's real T-O bond lengths if you get mis-detected bonds.
* If many T atoms are reported as still over-hydroxylated after rescuing,
  try a larger --wiggle, or accept that a fully strict cylinder isn't
  compatible with your --max-caps limit at this diameter.

-------------------------------------------------------------------------
EXAMPLES
-------------------------------------------------------------------------
  # 20 A diameter pore along z, up to 2 caps per Si/Al, +/-1.5 A wiggle
  python3 mesopore_carve.py CONFIG CONFIG_pore --diameter 20

  # pore along an arbitrary diagonal direction, gro -> gro, bigger wiggle
  python3 mesopore_carve.py zeolite.gro zeolite_pore.gro \
      --diameter 15 --axis 1 1 0 --center 20 20 15 --wiggle 2.0

  # framework contains Ge as well as Si/Al, allow up to 3 caps per T atom
  python3 mesopore_carve.py CONFIG CONFIG_pore --diameter 18 \
      --tcations Si Al Ge --cutoff 2.1 --max-caps 3
"""

import argparse
from itertools import product

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Element guessing
# ---------------------------------------------------------------------------

KNOWN_ELEMENTS = [
    "Si", "Al", "Ga", "Ge", "Fe", "Ti", "Zn", "Na", "Ca", "Mg", "K",
    "P", "B", "O", "H", "C", "N", "Sn", "Zr",
]


def guess_element(name):
    """Guess an element symbol from an atom name/label like 'Si1', 'OB2', 'O'."""
    letters = "".join(c for c in name if c.isalpha())
    for length in (2, 1):
        cand = letters[:length].capitalize()
        if cand in KNOWN_ELEMENTS:
            return cand
    return letters[:1].capitalize() if letters else "X"


# ---------------------------------------------------------------------------
# I/O: GROMACS .gro
# ---------------------------------------------------------------------------

def parse_gro(path):
    with open(path) as f:
        lines = f.readlines()
    title = lines[0].rstrip("\n")
    natoms = int(lines[1].split()[0])
    atom_lines = lines[2:2 + natoms]

    resnums, resnames, names, coords = [], [], [], []
    for l in atom_lines:
        resnums.append(int(l[0:5]))
        resnames.append(l[5:10].strip())
        names.append(l[10:15].strip())
        x = float(l[20:28]); y = float(l[28:36]); z = float(l[36:44])
        coords.append((x, y, z))
    coords = np.array(coords) * 10.0  # nm -> Angstrom

    boxvals = [float(v) for v in lines[2 + natoms].split()]
    cell = np.zeros((3, 3))
    cell[0, 0], cell[1, 1], cell[2, 2] = boxvals[0], boxvals[1], boxvals[2]
    if len(boxvals) == 9:
        cell[0, 1], cell[0, 2] = boxvals[3], boxvals[4]
        cell[1, 0], cell[1, 2] = boxvals[5], boxvals[6]
        cell[2, 0], cell[2, 1] = boxvals[7], boxvals[8]
    cell *= 10.0  # nm -> Angstrom

    return {
        "title": title, "resnums": resnums, "resnames": resnames,
        "names": names, "coords": coords, "cell": cell, "format": "gro",
    }


def write_gro(path, data):
    coords = data["coords"] / 10.0
    cell = data["cell"] / 10.0
    n = len(coords)
    with open(path, "w") as f:
        f.write(data.get("title", "carved structure") + "\n")
        f.write(f"{n}\n")
        for i in range(n):
            resnum = data["resnums"][i] if i < len(data["resnums"]) else 1
            resname = data["resnames"][i] if i < len(data["resnames"]) else "ZEO"
            name = data["names"][i]
            x, y, z = coords[i]
            f.write(
                f"{resnum % 100000:5d}{resname:<5.5s}{name:>5.5s}"
                f"{(i + 1) % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n"
            )
        v1x, v2y, v3z = cell[0, 0], cell[1, 1], cell[2, 2]
        v1y, v1z = cell[0, 1], cell[0, 2]
        v2x, v2z = cell[1, 0], cell[1, 2]
        v3x, v3y = cell[2, 0], cell[2, 1]
        if any(abs(v) > 1e-6 for v in (v1y, v1z, v2x, v2z, v3x, v3y)):
            f.write(
                f"{v1x:.5f} {v2y:.5f} {v3z:.5f} {v1y:.5f} {v1z:.5f} "
                f"{v2x:.5f} {v2z:.5f} {v3x:.5f} {v3y:.5f}\n"
            )
        else:
            f.write(f"{v1x:.5f} {v2y:.5f} {v3z:.5f}\n")


# ---------------------------------------------------------------------------
# I/O: DL_POLY CONFIG
# ---------------------------------------------------------------------------

def parse_config(path):
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    title = lines[0]
    header = lines[1].split()
    levcfg, imcon = int(header[0]), int(header[1])

    idx = 2
    cell = np.zeros((3, 3))
    if imcon > 0:
        for i in range(3):
            cell[i] = [float(v) for v in lines[idx].split()]
            idx += 1

    names, coords = [], []
    n = len(lines)
    while idx < n:
        line = lines[idx].strip()
        if line == "":
            idx += 1
            continue
        name = line.split()[0]
        idx += 1
        x, y, z = (float(v) for v in lines[idx].split()[:3])
        idx += 1
        idx += levcfg  # skip velocity / force lines if present
        names.append(name)
        coords.append((x, y, z))
    coords = np.array(coords)

    return {
        "title": title, "levcfg": levcfg, "imcon": imcon,
        "names": names, "coords": coords, "cell": cell, "format": "config",
    }


def write_config(path, data):
    cell = data["cell"]
    coords = data["coords"]
    names = data["names"]
    levcfg = 0
    imcon = data.get("imcon", 3) if np.any(cell) else 0
    with open(path, "w") as f:
        f.write(data.get("title", "carved structure") + "\n")
        f.write(f"{levcfg:10d}{imcon:10d}{len(names):10d}\n")
        if imcon > 0:
            for i in range(3):
                f.write(
                    f"{cell[i, 0]:20.10f}{cell[i, 1]:20.10f}{cell[i, 2]:20.10f}\n"
                )
        for i, (nm, (x, y, z)) in enumerate(zip(names, coords)):
            f.write(f"{nm:<8s}{i + 1:10d}\n")
            f.write(f"{x:20.10f}{y:20.10f}{z:20.10f}\n")


# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------

def load_structure(path):
    if path.lower().endswith(".gro"):
        return parse_gro(path)
    try:
        with open(path) as f:
            lines = f.readlines()
        header = lines[1].split()
        int(header[0]); int(header[1])
        return parse_config(path)
    except Exception:
        return parse_gro(path)


def save_structure(path, data):
    if path.lower().endswith(".gro"):
        write_gro(path, data)
    else:
        write_config(path, data)


# ---------------------------------------------------------------------------
# Bonding (periodic, T-O only)
# ---------------------------------------------------------------------------

def build_bonds(coords, elements, cell, cutoffs, tcations):
    n = len(coords)
    bonds = [[] for _ in range(n)]
    t_idx = [i for i, e in enumerate(elements) if e in tcations]
    o_idx = [i for i, e in enumerate(elements) if e == "O"]
    if not t_idx or not o_idx:
        return bonds

    if np.any(cell):
        shifts = np.array(list(product([-1, 0, 1], repeat=3)))
        cell_shifts = shifts @ cell
    else:
        cell_shifts = np.zeros((1, 3))

    o_pos = coords[o_idx]
    exp_pos, exp_map = [], []
    for shift in cell_shifts:
        exp_pos.append(o_pos + shift)
        exp_map.extend(o_idx)
    exp_pos = np.vstack(exp_pos)
    tree = cKDTree(exp_pos)

    max_cut = max(cutoffs.values())
    for ti in t_idx:
        cut = cutoffs.get(elements[ti], max_cut)
        for j in tree.query_ball_point(coords[ti], r=cut):
            oi = exp_map[j]
            d = np.linalg.norm(coords[ti] - exp_pos[j])
            if d < cut and oi not in bonds[ti]:
                bonds[ti].append(oi)
                if ti not in bonds[oi]:
                    bonds[oi].append(ti)
    return bonds


# ---------------------------------------------------------------------------
# Existing O-H detection (e.g. Bronsted acid O_b-H_b sites already present
# in the input structure, which must NOT be touched by silanol capping)
# ---------------------------------------------------------------------------

def find_existing_oh(coords, elements, cell, cutoff=1.3):
    """Return a boolean array (len = n atoms) that is True for O atoms which
    already have a bonded H in the ORIGINAL structure, before any pore
    carving. These sites must be left untouched by the capping step."""
    n = len(coords)
    has_h = np.zeros(n, dtype=bool)
    o_idx = [i for i, e in enumerate(elements) if e == "O"]
    h_idx = [i for i, e in enumerate(elements) if e == "H"]
    if not o_idx or not h_idx:
        return has_h

    if np.any(cell):
        shifts = np.array(list(product([-1, 0, 1], repeat=3)))
        cell_shifts = shifts @ cell
    else:
        cell_shifts = np.zeros((1, 3))

    h_pos = coords[h_idx]
    exp_pos = []
    for shift in cell_shifts:
        exp_pos.append(h_pos + shift)
    exp_pos = np.vstack(exp_pos)
    tree = cKDTree(exp_pos)

    for oi in o_idx:
        if tree.query_ball_point(coords[oi], r=cutoff):
            has_h[oi] = True
    return has_h


# ---------------------------------------------------------------------------
# Radial distance from the pore axis (periodic-aware)
# ---------------------------------------------------------------------------

def compute_radial(coords, center, axis, cell):
    """For every atom, compute its perpendicular distance from the pore
    axis (a line through `center` along `axis`), using the minimum image
    convention so this is correct even near cell boundaries. Also returns
    the minimum-image Cartesian coordinates actually used."""
    axis = axis / np.linalg.norm(axis)

    if np.any(cell):
        inv_cell = np.linalg.inv(cell)
        rel = coords - center
        frac = rel @ inv_cell
        frac -= np.round(frac)          # minimum image w.r.t. pore centre
        mi_coords = center + frac @ cell
    else:
        mi_coords = coords.copy()

    rel = mi_coords - center
    along = np.outer(rel @ axis, axis)
    perp = rel - along
    radial = np.linalg.norm(perp, axis=1)
    return radial, mi_coords


# ---------------------------------------------------------------------------
# Hydroxylation-aware keep/remove decision
# ---------------------------------------------------------------------------

def count_caps(i, elements, bonds, keep_mask):
    """Number of capping (-OH) groups T atom i would end up with, given the
    current keep_mask. Returns (count, list_of_fix_sets) where each entry
    in list_of_fix_sets is the set of atom indices that would ALL need to
    be rescued (kept) to eliminate that one particular cap."""
    fixes = []
    for o in bonds[i]:
        if elements[o] != "O":
            continue
        if not keep_mask[o]:
            fixes.append({o})
            continue
        other_T = [k for k in bonds[o] if elements[k] != "O" and k != i]
        missing = [k for k in other_T if not keep_mask[k]]
        if missing:
            fixes.append(set(missing))
    return len(fixes), fixes


def decide_keep_mask(elements, bonds, radial, radius, wiggle, tcations,
                      max_caps=2, max_iter=200):
    """
    Decide which atoms to actually remove, starting from a plain cylinder
    cut, then rescuing (un-removing) atoms - ONLY within +/-`wiggle` A of
    the nominal pore wall - wherever doing so is needed to keep every
    surviving T atom at or below `max_caps` capping groups.
    """
    n = len(radial)
    keep_mask = radial >= radius                       # nominal cylinder cut
    flexible = np.abs(radial - radius) <= wiggle        # atoms allowed to move

    t_atoms = [i for i, e in enumerate(elements) if e in tcations]

    print(f"Hydroxylation-aware carving: wiggle = +/-{wiggle:.2f} A, "
          f"max {max_caps} capping group(s) per T atom.")

    for iteration in range(max_iter):
        changed = False
        for i in t_atoms:
            if not keep_mask[i]:
                continue
            n_caps, fixes = count_caps(i, elements, bonds, keep_mask)
            excess = n_caps - max_caps
            if excess <= 0:
                continue
            # try the smallest rescues first (usually a single atom)
            fixes.sort(key=len)
            n_fixed = 0
            for needed in fixes:
                if n_fixed >= excess:
                    break
                if all(flexible[a] and not keep_mask[a] for a in needed):
                    for a in needed:
                        keep_mask[a] = True
                    n_fixed += 1
                    changed = True
        if not changed:
            break
    else:
        print(f"WARNING: hydroxylation rescue did not fully converge after "
              f"{max_iter} iterations - results below may still include "
              f"fixable over-hydroxylation.")

    n_rescued = int(np.sum(flexible & keep_mask & (radial < radius)))
    print(f"Rescued {n_rescued} atom(s) inside the nominal radius to avoid "
          f"over-hydroxylation (wiggle-limited pass).")

    return keep_mask


def restore_fully_bridged_oxygens(elements, bonds, keep_mask, tcations):
    """
    A bridging O connects exactly two T atoms. If the O is marked for
    removal but BOTH of its original T neighbours survive, restore it
    (unconditionally - not limited by --wiggle). See "SHARED-OXYGEN
    CLEANUP" in the module docstring for the reasoning: leaving it removed
    would make two different T atoms independently place a new capping
    -OH at the same now-empty point in space.
    """
    n = len(keep_mask)
    n_restored = 0
    for o in range(n):
        if elements[o] != "O" or keep_mask[o]:
            continue
        t_neighbours = [j for j in bonds[o] if elements[j] in tcations]
        if len(t_neighbours) >= 2 and all(keep_mask[j] for j in t_neighbours):
            keep_mask[o] = True
            n_restored += 1
    print(f"Restored {n_restored} bridging O atom(s) whose T neighbours "
          f"both survive (avoids duplicate/overlapping capping groups).")
    return keep_mask


def report_final_hydroxylation(elements, bonds, keep_mask, tcations, max_caps):
    """Final, authoritative check (after all rescue/cleanup steps) of how
    many capping groups every surviving T atom actually ends up with."""
    residual = []
    for i, e in enumerate(elements):
        if e not in tcations or not keep_mask[i]:
            continue
        n_caps, _ = count_caps(i, elements, bonds, keep_mask)
        if n_caps > max_caps:
            residual.append((i, n_caps))

    if residual:
        print(f"WARNING: {len(residual)} T atom(s) STILL exceed {max_caps} "
              f"capping group(s) even after all rescues. Try a larger "
              f"--wiggle, or accept these as-is.")
        for i, c in residual[:20]:
            print(f"    atom index {i} ({elements[i]}): {c} capping groups")
        if len(residual) > 20:
            print(f"    ... and {len(residual) - 20} more")
    else:
        print(f"Final check: all surviving T atoms are at or below "
              f"{max_caps} capping group(s).")
    return residual


# ---------------------------------------------------------------------------
# Silanol capping
# ---------------------------------------------------------------------------

def cap_silanols(keep_mask, coords, elements, names, bonds, tcations,
                  has_existing_h, oh_bond=0.98, to_bond=1.62):
    n = len(coords)
    new_coords, new_elements, new_names = [], [], []
    kept_old_to_new = {}
    for i in range(n):
        if keep_mask[i]:
            kept_old_to_new[i] = len(new_coords)
            new_coords.append(coords[i])
            new_elements.append(elements[i])
            new_names.append(names[i])

    added_h = 0
    added_oh_group = 0
    skipped_existing_oh = 0

    # (a) surviving bridging O that lost >=1 T neighbour -> add H.
    # Direction: the REAL original vector from this O towards where its
    # missing T neighbour(s) used to sit - reusing the true local
    # geometry rather than guessing, so the O-H bond points the way the
    # broken T-O bond really did.
    for i in range(n):
        if not keep_mask[i] or elements[i] != "O":
            continue
        if has_existing_h[i]:
            # Already carries an H in the original structure (e.g. a
            # Bronsted-acid O_b-H_b site). Leave it exactly as it is,
            # even if it has now become terminal.
            skipped_existing_oh += 1
            continue
        original_T = [j for j in bonds[i] if elements[j] in tcations]
        remaining_T = [j for j in original_T if keep_mask[j]]
        missing_T = [j for j in original_T if not keep_mask[j]]
        if remaining_T and missing_T:
            o_pos = coords[i]
            missing_avg = np.mean([coords[j] for j in missing_T], axis=0)
            direction = missing_avg - o_pos   # real bond direction, O -> missing T
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
            h_pos = o_pos + direction * oh_bond
            new_coords.append(h_pos)
            new_elements.append("H")
            new_names.append("H_s")
            new_names[kept_old_to_new[i]] = "O_s"  # relabel the surviving O
            added_h += 1

    # (b) surviving T atom that lost an O ENTIRELY -> add a new -OH group
    # for EACH missing O, individually, along the REAL original T-O bond
    # vector for that specific oxygen. Because restore_fully_bridged_
    # oxygens() has already run, every missing O here is guaranteed to
    # have had exactly one surviving T neighbour (this one), so there is
    # no risk of a different T atom also claiming the same position.
    # Bond DIRECTION is real/original; bond LENGTH is standardised to
    # to_bond/oh_bond (see docstring caveats).
    for i in range(n):
        if not keep_mask[i] or elements[i] not in tcations:
            continue
        original_O = [j for j in bonds[i] if elements[j] == "O"]
        missing_O = [j for j in original_O if not keep_mask[j]]
        if not missing_O:
            continue

        t_pos = coords[i]
        for o in missing_O:
            direction = coords[o] - t_pos   # real original T-O bond vector
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
            o_pos = t_pos + direction * to_bond
            h_pos = o_pos + direction * oh_bond
            new_coords.append(o_pos); new_elements.append("O"); new_names.append("O_s")
            new_coords.append(h_pos); new_elements.append("H"); new_names.append("H_s")
            added_oh_group += 1

    return (np.array(new_coords), new_elements, new_names,
            added_h, added_oh_group, skipped_existing_oh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Carve a cylindrical mesopore in a zeolite supercell "
                     "and cap dangling bonds as silanols.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Input structure file (CONFIG or .gro)")
    p.add_argument("output", help="Output structure file (CONFIG or .gro, by extension)")
    p.add_argument("--diameter", type=float, required=True, help="Pore diameter in Angstrom")
    p.add_argument("--center", type=float, nargs=3, default=None,
                   help="Pore centre x y z (Angstrom). Default: cell centroid")
    p.add_argument("--axis", type=float, nargs=3, default=[0, 0, 1],
                   help="Pore axis direction vector (any direction, need not be normalised). Default: 0 0 1")
    p.add_argument("--tcations", nargs="+", default=["Si", "Al"],
                   help="Framework tetrahedral cation elements. Default: Si Al")
    p.add_argument("--cutoff", type=float, default=2.0,
                   help="T-O bond distance cutoff in Angstrom (default 2.0)")
    p.add_argument("--oh-bond", type=float, default=0.98,
                   help="O-H bond length for new silanols (Angstrom)")
    p.add_argument("--to-bond", type=float, default=1.62,
                   help="T-O bond length for new silanol T-OH groups (Angstrom)")
    p.add_argument("--wiggle", type=float, default=1.5,
                   help="How far (Angstrom) inside/outside the nominal pore "
                        "wall an atom may be rescued from removal to avoid "
                        "over-hydroxylation (default 1.5)")
    p.add_argument("--max-caps", type=int, default=2,
                   help="Maximum number of capping -OH groups allowed on "
                        "any single T atom (default 2)")
    args = p.parse_args()

    data = load_structure(args.input)
    coords = data["coords"]
    names = data["names"]
    elements = [guess_element(nm) for nm in names]
    cell = data["cell"]

    center = (np.array(args.center) if args.center is not None
              else (cell.sum(axis=0) / 2.0 if np.any(cell) else coords.mean(axis=0)))
    axis = np.array(args.axis, dtype=float)

    cutoffs = {el: args.cutoff for el in args.tcations}
    print(f"Building framework bond list ({'/'.join(args.tcations)}-O, cutoff {args.cutoff} A)...")
    bonds = build_bonds(coords, elements, cell, cutoffs, args.tcations)

    print("Checking for pre-existing O-H sites (e.g. Bronsted O_b-H_b)...")
    has_existing_h = find_existing_oh(coords, elements, cell)

    radial, mi_coords = compute_radial(coords, center, axis, cell)
    print(f"Centre = {center}, axis = {axis / np.linalg.norm(axis)}")

    tcations_set = set(args.tcations)
    keep_mask = decide_keep_mask(
        elements, bonds, radial, args.diameter / 2.0, args.wiggle,
        tcations_set, max_caps=args.max_caps,
    )
    keep_mask = restore_fully_bridged_oxygens(elements, bonds, keep_mask, tcations_set)
    report_final_hydroxylation(elements, bonds, keep_mask, tcations_set, args.max_caps)

    print(f"Removing {np.sum(~keep_mask)} atoms (nominal diameter {args.diameter} A).")

    new_coords, new_elements, new_names, added_h, added_oh, skipped = cap_silanols(
        keep_mask, mi_coords, elements, names, bonds, tcations_set,
        has_existing_h, oh_bond=args.oh_bond, to_bond=args.to_bond,
    )

    print(f"Added {added_h} silanol H atoms to dangling bridging oxygens.")
    print(f"Added {added_oh} new -OH group(s) to undercoordinated T atoms.")
    print(f"Skipped {skipped} O atom(s) that already had an H (e.g. Bronsted sites) - left untouched.")
    print(f"Final atom count: {len(new_coords)} (was {len(coords)})")

    out_data = {
        "title": data.get("title", "mesoporous structure") + " (mesopore carved)",
        "coords": new_coords,
        "names": new_names,
        "cell": cell,
        "levcfg": 0,
        "imcon": data.get("imcon", 3),
        "resnums": [1] * len(new_coords),
        "resnames": ["ZEO"] * len(new_coords),
    }
    save_structure(args.output, out_data)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
