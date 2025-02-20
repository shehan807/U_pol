#!/usr/bin/env python3
"""
Example wrapper for:
  (1) Parsing SAPT Psi4 .out files
  (2) Writing new PDBs from a template
  (3) Running polarization.py
  (4) Extracting induction energies
  (5) Creating a scatter plot
"""

#!/usr/bin/env python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse
import shutil 

BOHR_TO_ANG = 0.52917721092

# ------------------------------------------------------------------------------
# 1. Load the base/template PDB
# ------------------------------------------------------------------------------

TEMPLATE_PDB = """\
REMARK LIGPARGEN GENERATED PDB FILE
ATOM      1  N00 IM      1      -2.608  -3.068   5.305
ATOM      2  N0  IM      1      -0.473  -2.926   1.720
ATOM      3  H2  IM      1       1.660  -6.359   2.220
ATOM      4  H21 IM      1      -1.000  -6.533   6.614
ATOM      5  H1  IM      1      -3.058  -0.006   2.756
ATOM      6  H0  IM      1       0.275  -2.247   0.000
ATOM      7  C2  IM      1       0.248  -5.075   2.941
ATOM      8  C21 IM      1      -1.084  -5.146   5.135
ATOM      9  C1  IM      1      -2.174  -1.777   3.211
ATOM     10  N00 IM      2      -3.044   3.972  -4.693
ATOM     11  N0  IM      2      -5.179   3.830  -1.109
ATOM     12  H2  IM      2      -7.312   7.262  -1.609
ATOM     13  H21 IM      2      -4.652   7.436  -6.003
ATOM     14  H1  IM      2      -2.594   0.909  -2.144
ATOM     15  H0  IM      2      -5.927   3.151   0.612
ATOM     16  C2  IM      2      -5.900   5.978  -2.329
ATOM     17  C21 IM      2      -4.568   6.049  -4.523
ATOM     18  C1  IM      2      -3.478   2.681  -2.599
TER
CONECT    8    7
CONECT    7    2
CONECT    2    6
CONECT    2    9
CONECT    8    1
CONECT    8    4
CONECT    7    3
CONECT    9    5
CONECT    9    1
CONECT   17   16
CONECT   16   11
CONECT   11   15
CONECT   11   18
CONECT   17   10
CONECT   17   13
CONECT   16   12
CONECT   18   14
CONECT   18   10
CONECT   26   25
CONECT   25   20
CONECT   20   24
CONECT   20   27
CONECT   26   19
CONECT   26   22
CONECT   25   21
CONECT   27   23
CONECT   27   19
END
"""

template_lines = [l for l in TEMPLATE_PDB.splitlines()]
#!/usr/bin/env python3
import os
import re
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Example constants
BOHR_TO_ANG = 0.529177210903

# 1) parse_sapt_outfile: ensures we only read the *first* Dimer HF geometry
def parse_sapt_outfile(outfile_path):
    """
    Parse the *first* Dimer HF geometry (in Bohr) and the induction energy (kJ/mol)
    from a Psi4 SAPT0 .out file.  We stop reading as soon as we detect the
    end of the first Dimer HF block (so we won't read monomer blocks).
    """

    coords_bohr = []
    induction_kjmol = None

    with open(outfile_path, "r") as f:
        lines = f.readlines()

    found_dimer_block = False
    reading_coords = False

    # -- Part A: read the first Dimer HF geometry --
    for line in lines:
        # Identify the Dimer HF block
        if not found_dimer_block and "Dimer HF" in line:
            found_dimer_block = True
            continue

        # Once we've found "Dimer HF," look for "Geometry (in Bohr)" to start reading
        if found_dimer_block and not reading_coords and "Geometry (in Bohr)" in line:
            reading_coords = True
            continue

        # If reading coordinates, watch for lines that signify the geometry block ended
        if reading_coords:
            lower_line = line.lower()
            if ("monomer" in lower_line and "hf" in lower_line) \
               or "running in" in lower_line \
               or "nuclear repulsion" in lower_line \
               or "rotational constants" in line \
               or "symmetry" in lower_line \
               or "set {" in lower_line \
               or "psi4:" in lower_line:
                # End of dimer geometry
                break

            # Skip blank lines, table headers, ghost atoms
            if not line.strip():
                continue
            if line.strip().startswith("Center") or line.strip().startswith("---") or "Gh(" in line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                coords_bohr.append((x, y, z))
            except ValueError:
                pass

    # -- Part B: parse induction from the "SAPT Results" section --
    sapt_block_found = False
    for line in lines:
        if "SAPT Results" in line:
            sapt_block_found = True
            continue
        if sapt_block_found and line.strip().startswith("Induction"):
            # Typically: Induction  -0.00016301 [mEh]  -0.00010229 [kcal/mol]  -0.00042799 [kJ/mol]
            parts = line.split()
            if len(parts) >= 6 and parts[-1] == "[kJ/mol]":
                try:
                    induction_kjmol = float(parts[-2])
                except ValueError:
                    pass
            break

    return coords_bohr, induction_kjmol

# 2) Minimal-rounding PDB writer
def write_pdb_from_template(coords_ang, template_lines, out_pdb_path):
    """
    Writes coords with minimal rounding (8 decimals). Adjust as needed.
    coords_ang: list of (x,y,z) in Angstrom
    template_lines: lines from a base PDB
    out_pdb_path: path to write PDB
    """
    out_lines = []
    atom_count = 0
    for line in template_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            x, y, z = coords_ang[atom_count]
            atom_count += 1
            # We'll slice up to column 30, then insert coords with 8 decimal places
            new_line = (
                f"{line[0:30]}"
                f"{x:8.3f}"
                f"{y:8.3f}"
                f"{z:8.3f}"
                f"{line[54:]}"
            )
            out_lines.append(new_line)
        else:
            out_lines.append(line)
    with open(out_pdb_path, "w") as fo:
        fo.write("\n".join(out_lines) + "\n")

# 3) parse_python_log, a placeholder that returns a float if found
import os

def parse_python_log(log_path):
    """
    Reads log.out to find a line matching:
      U_ind = <float> kJ/mol
    Returns that float. Then deletes log_path so the next run can create a fresh log.
    """
    val = None
    if not os.path.exists(log_path):
        return val

    with open(log_path, "r") as f:
        end_file = False
        for line in f:
            if "JAXOPT.BFGS Minimizer completed" in line:
                end_file = True
                continue
            if end_file:
                if "U_ind =" in line and "kJ/mol" in line:
                    parts = line.split()
                    # e.g. ["U_ind", "=", "-10.941831...", "kJ/mol"]
                    try:
                        val = float(parts[2])
                    except ValueError:
                        pass
                    break

    # Delete old log once we've parsed the needed info
    try:
        os.remove(log_path)
    except Exception as e:
        print(f"Warning: could not remove log file {log_path}. Error: {e}")
    return val

# Suppose we have as many ATOM lines as you expect from the Dimer geometry
pdb_atom_lines = [ln for ln in template_lines if ln.startswith("ATOM") or ln.startswith("HETATM")]

def main():
    parser = argparse.ArgumentParser(description="Wrapper for SAPT DFT vs. MD induction energies.")
    parser.add_argument("--indir", type=str, default=".",
                        help="Directory with .out files to parse")
    parser.add_argument("--poldir", type=str, default=".",
                        help="Directory where polarization.py is located (if needed).")
    parser.add_argument("--no-run", action="store_true",
                        help="If set, skip calling polarization.py; parse existing log.outs only.")
    args = parser.parse_args()

    outfiles = sorted([f for f in os.listdir(args.indir) if f.endswith(".out")])

    sapt_energies = []
    python_energies = []
    dimer_ids = []  # We will color by this integer ID

    pdb_dir = os.path.join(args.indir, "generated_pdbs")
    os.makedirs(pdb_dir, exist_ok=True)

    for outfn in outfiles:
        # Attempt to parse ID from filename, e.g. '2mer-0+123.out' => 123
        match = re.search(r"\+(\d+)\.out$", outfn)
        if not match:
            # If no match, skip or set to some default
            print(f"Could not parse ID from {outfn}, skipping.")
            continue
        dimer_id = int(match.group(1))

        out_path = os.path.join(args.indir, outfn)
        print(f"Parsing {out_path}")

        coords_bohr, sapt_E_ind_kj = parse_sapt_outfile(out_path)
        if sapt_E_ind_kj is None or not coords_bohr:
            print(f"  -> Could not parse geometry or induction from {outfn}, skipping.")
            continue

        # Convert geometry to Å
        coords_ang = [(x_b * BOHR_TO_ANG, y_b * BOHR_TO_ANG, z_b * BOHR_TO_ANG) for (x_b,y_b,z_b) in coords_bohr]

        # Check # of coords vs # of ATOM lines in the template
        if len(coords_ang) != len(pdb_atom_lines):
            print(f"  -> WARNING: mismatch in # of parsed atoms vs. template lines. Skipping.")
            continue

        # Write new PDB
        base_name = os.path.splitext(outfn)[0]
        new_pdb_name = f"{base_name}.pdb"
        new_pdb_path = os.path.join(pdb_dir, new_pdb_name)
        write_pdb_from_template(coords_ang, template_lines, new_pdb_path)

        # Optionally run polarization.py
        if not args.no_run:
            imidazole3 = "/storage/home/hcoda1/4/sparmar32/p-jmcdaniel43-0/scripts/U_pol/benchmarks/OpenMM/imidazole3"
            shutil.copyfile(new_pdb_path, os.path.join(imidazole3, "imidazole3.pdb"))
            cmd = [
                "python",
                os.path.join(args.poldir, "polarization.py"),
                "--mol", "imidazole3",
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

        # Parse Python induction from log.out
        pol_log = os.path.join(args.poldir, "log.out")
        python_E_ind_kj = parse_python_log(pol_log)

        if python_E_ind_kj is not None:
            print(f"  SAPT induction = {sapt_E_ind_kj:.6f} kJ/mol, "
                  f"Python induction = {python_E_ind_kj:.6f} kJ/mol, "
                  f"Dimer ID = {dimer_id}")
            sapt_energies.append(sapt_E_ind_kj)
            python_energies.append(python_E_ind_kj)
            dimer_ids.append(int(dimer_id))
        else:
            print("  Could not parse python induction from log.out.")

    # Now plot, color-coded by dimer_ids
    if len(sapt_energies) > 0:
        sapt_energies = np.array(sapt_energies)
        python_energies = np.array(python_energies)
        dimer_ids = np.array(dimer_ids)

        plt.figure(figsize=(6,6))
        sc = plt.scatter(sapt_energies, python_energies, c=dimer_ids,
                         cmap='viridis', alpha=0.8, edgecolors='k')

        # y=x line
        min_val = min(sapt_energies.min(), python_energies.min()) * 1.05
        max_val = max(sapt_energies.max(), python_energies.max()) * 1.05
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        plt.xlabel("SAPT Induction (kJ/mol)")
        plt.ylabel("MD/Python Induction (kJ/mol)")
        plt.title("Induction Energy Comparison")
        plt.grid(False)
        plt.legend()

        # colorbar
        cb = plt.colorbar(sc)
        cb.set_label("dimer distance (ID)")

        plt.tight_layout()
        plt.savefig("sapt_vs_python_induction.png", dpi=150)
        plt.show()

        np.savetxt("induction_data.csv",
                   np.column_stack([sapt_energies, python_energies, dimer_ids]),
                   delimiter=",",
                   header="SAPT_Induction,MD_Induction,DimerID",
                   comments="")

    else:
        print("No induction energies collected; nothing to plot.")

if __name__ == "__main__":
    main()

