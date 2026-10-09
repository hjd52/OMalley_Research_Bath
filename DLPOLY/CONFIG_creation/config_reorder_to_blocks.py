#!/usr/bin/env python3

def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def read_config(filename):
    with open(filename, "r") as f:
        all_lines = [l.rstrip() for l in f if l.strip() != ""]

    # ---- HEADER ----
    header = all_lines[:5]          # preserve exactly
    body_lines = all_lines[5:]      # atoms start here

    # levcfg is on line 2 (0-based index 1), first integer
    levcfg = int(header[1].split()[0])

    # ---- TOKENISE BODY ----
    tokens = []
    for line in body_lines:
        tokens.extend(line.split())

    atoms = []
    i = 0
    n = len(tokens)

    while i < n:
        # Atom label
        label = tokens[i]
        i += 1

        # Optional atom index (ignore)
        if i < n and tokens[i].isdigit():
            i += 1

        # Coordinates: exactly 3 floats
        coords = []
        while len(coords) < 3:
            if i >= n or not is_float(tokens[i]):
                raise ValueError(
                    f"Failed to parse coordinates for atom '{label}' "
                    f"(token='{tokens[i] if i < n else 'EOF'}')"
                )
            coords.append(float(tokens[i]))
            i += 1

        # Velocities if levcfg > 0
        vel = None
        if levcfg > 0:
            vel = []
            while len(vel) < 3:
                if i >= n or not is_float(tokens[i]):
                    raise ValueError(
                        f"Failed to parse velocities for atom '{label}' "
                        f"(token='{tokens[i] if i < n else 'EOF'}')"
                    )
                vel.append(float(tokens[i]))
                i += 1

        atoms.append({
            "label": label,
            "coords": coords,
            "vel": vel
        })

    return header, atoms


def reorder_atoms(atoms):
    """
    Reorders atoms into the groups:
        O (framework), Si, Al, O_b, H_b, <everything else>

    Al-O-H triplets (in-order, consecutive) are converted to
    Al / O_b / H_b just like in the original triplet script.

    Any atom label that isn't O, Si, or part of an Al-O-H triplet
    is left untouched and appended at the very end, in its original
    relative order (not raised as an error).
    """
    framework_O = []
    Si = []
    Al = []
    O_b = []
    H_b = []
    others = []

    used = [False] * len(atoms)
    i = 0

    # Detect Al-O-H triplets in-order
    while i < len(atoms) - 2:
        a1, a2, a3 = atoms[i], atoms[i+1], atoms[i+2]

        if a1["label"] == "Al" and a2["label"] == "O" and a3["label"] == "H":
            Al.append(a1)

            ob = dict(a2)
            ob["label"] = "O_b"
            O_b.append(ob)

            hb = dict(a3)
            hb["label"] = "H_b"
            H_b.append(hb)

            used[i] = used[i+1] = used[i+2] = True
            i += 3
        else:
            i += 1

    # Remaining atoms
    for idx, atom in enumerate(atoms):
        if used[idx]:
            continue

        if atom["label"] == "O":
            framework_O.append(atom)
        elif atom["label"] == "Si":
            Si.append(atom)
        else:
            # Instead of raising, just stash it for the tail of the file
            others.append(atom)

    if not (len(Al) == len(O_b) == len(H_b)):
        raise ValueError("Triplet mismatch: Al, O_b, H_b counts differ")

    return framework_O + Si + Al + O_b + H_b + others


def write_config(filename, header, atoms):
    with open(filename, "w") as f:
        for line in header:
            f.write(f"{line}\n")

        for idx, atom in enumerate(atoms, start=1):
            f.write(f"{atom['label']} {idx}\n")
            f.write("{:20.10f}{:20.10f}{:20.10f}\n".format(*atom["coords"]))
            if atom["vel"] is not None:
                f.write("{:20.10f}{:20.10f}{:20.10f}\n".format(*atom["vel"]))


def main():
    infile = "CONFIG"
    outfile = "CONFIG_reordered"

    header, atoms = read_config(infile)
    new_atoms = reorder_atoms(atoms)
    write_config(outfile, header, new_atoms)

    print("Reordering complete.")
    print(f"Total atoms: {len(new_atoms)}")
    print(f"O   : {sum(a['label']=='O' for a in new_atoms)}")
    print(f"Si  : {sum(a['label']=='Si' for a in new_atoms)}")
    print(f"Al  : {sum(a['label']=='Al' for a in new_atoms)}")
    print(f"O_b : {sum(a['label']=='O_b' for a in new_atoms)}")
    print(f"H_b : {sum(a['label']=='H_b' for a in new_atoms)}")

    other_labels = sorted({a['label'] for a in new_atoms}
                           - {'O', 'Si', 'Al', 'O_b', 'H_b'})
    if other_labels:
        print("Other labels kept at end:")
        for lbl in other_labels:
            print(f"  {lbl} : {sum(a['label']==lbl for a in new_atoms)}")


if __name__ == "__main__":
    main()
