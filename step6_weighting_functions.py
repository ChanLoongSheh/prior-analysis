# (S6) Step 6 — Generate PCA-based Radiance Perturbation Spectra (Weighting Functions)
# Title: step6_weighting_functions.py
# Author: (Your Lab)
# Description:
#   Implements Step 6 from the project proposal (README.md): For each joint PCA perturbation
#   (±Δc_i along i=1..9), compute the radiance-difference spectrum in the wavelength domain:
#       ΔI_i,±(λ) = I_pert_i,±(λ) − I_ref(λ)
#   using the provided .mat files. Save outputs (CSVs with index and column headers), produce
#   18 plots (one per perturbation), and write a detailed inventory log explaining all files.
#
# Notes:
#   - This step uses the "rad_um" spectra and "lambda" grid (λ in micrometers).
#   - Units retained: W / (m^2 · sr · μm). Sign is retained, as required.
#   - No detector system effects, no filter transmittances, no ZnSe, no LiTaO3 emission here.
#   - No command-line parsing; outputs go to fixed directory without timestamps.
#
# How to run (main program):
#   1) Edit USER PATHS below to match your environment.
#   2) Run: run_step6_weighting_functions_main()
#
# Key variables (readable, with meanings):
#   - λ_ref (shape: [Nλ,]): wavelength grid in micrometers (2.5–20 μm). Symbol: lambda in code as lam_ref.
#   - I_ref(λ) (shape: [Nλ,]): representative downwelling spectral radiance in wavelength domain
#       Units: W/(m^2 · sr · μm). Symbol: rad_um_ref.
#   - I_pert_i,±(λ) (shape: [Nλ,]): perturbed downwelling spectral radiance in wavelength domain
#       for PC i and sign ±Δc_i, from 18 provided .mat files. Symbol: rad_um_pert.
#   - ΔI_i,±(λ) = I_pert_i,±(λ) − I_ref(λ)  (shape: [Nλ,]): weighting function spectra to be saved and plotted.
#       Units: W/(m^2 · sr · μm). Symbol: delta_I.
#   - Mapping of file index to (PC, sign): indices 1..9 -> +Δc1..+Δc9; indices 10..18 -> −Δc1..−Δc9.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch runs
import matplotlib.pyplot as plt

# Optional: robust loading for both MATLAB v7 and v7.3 files
from scipy.io import loadmat
import h5py


# =========================
# USER PATHS (edit here)
# =========================
DATA_ROOT = Path(r"C:\Users\24573\OneDrive - The University of Hong Kong\PhD\Project\atmospheric sounding\Data\DeltaC_TDp_change_forChenglong")

REPRESENTATIVE_FILENAME = "Rad_NewModel66layers_TDpP4Aerosol_HKObservatory0-100-5000m_6-1-15km_280.mat"
PERTURBATION_PATTERN = "Rad_DeltaC_TDp_index{idx}.mat"  # idx ∈ [1..18]

# Output root (no timestamp)
OUT_DIR = Path("./Step6_PCA_WeightingFunctions")

# Plot style
plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


# (S6.1) Function — Title: Safe loader for MATLAB files (supports v7 and v7.3)
def load_mat_file_safe(path: Path) -> Dict[str, np.ndarray]:
    """
    Attempts to load a MATLAB .mat file. First tries scipy.io.loadmat (v7),
    if that fails (e.g., v7.3), falls back to h5py.
    Returns a dict of arrays where entries are squeezed to 1D where applicable.
    """
    try:
        mdict = loadmat(str(path))
        # Remove meta-keys added by scipy
        out = {}
        for k, v in mdict.items():
            if k.startswith("__"):
                continue
            arr = np.array(v)
            out[k] = np.squeeze(arr)
        return out
    except Exception:
        # v7.3 (HDF5) case
        out = {}
        with h5py.File(str(path), "r") as f:
            for k in f.keys():
                ds = f[k]
                if isinstance(ds, h5py.Dataset):
                    arr = np.array(ds)
                    out[k] = np.squeeze(arr)
                else:
                    # handle nested groups by flattening basic datasets
                    # (assumes no deep nesting for these files)
                    # If needed, extend to recursive traversal
                    pass
        return out


# (S6.2) Function — Title: Extract wavelength-domain radiance and wavelength grid from dict
def extract_lambda_and_radiance(mdict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts:
      - lam (λ): wavelength grid in micrometers (μm) — mdict["lambda"]
      - rad_um: radiance in wavelength domain — mdict["rad_um"]
    Enforces 1D shapes and ascending λ. If λ is descending, both arrays are flipped.
    """
    # MATLAB variable names expected:
    #   "lambda" -> λ [μm]
    #   "rad_um" -> I(λ) [W/(m^2 · sr · μm)]
    if "lambda" not in mdict or "rad_um" not in mdict:
        raise KeyError("Required keys 'lambda' and/or 'rad_um' not found in .mat file.")

    lam = np.ravel(mdict["lambda"]).astype(float)
    rad_um = np.ravel(mdict["rad_um"]).astype(float)

    if lam.shape != rad_um.shape:
        raise ValueError(f"lambda and rad_um have mismatched shapes: {lam.shape} vs {rad_um.shape}")

    # Ensure ascending λ
    if lam[0] > lam[-1]:
        lam = lam[::-1]
        rad_um = rad_um[::-1]

    return lam, rad_um


# (S6.3) Function — Title: Map perturbation index (1..18) to (PC, sign)
def index_to_pc_sign(idx: int) -> Tuple[int, str]:
    """
    Maps the file index to (pc, sign):
      - 1..9   -> (pc=idx, sign="+")
      - 10..18 -> (pc=idx-9, sign="-")
    """
    if 1 <= idx <= 9:
        return idx, "+"
    elif 10 <= idx <= 18:
        return idx - 9, "-"
    else:
        raise ValueError("Index must be in [1..18].")


# (S6.4) Function — Title: Load representative spectrum I_ref(λ)
def load_reference_spectrum(rep_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the representative .mat file and returns:
      - lam_ref (λ_ref): wavelength grid [μm]
      - rad_um_ref: representative radiance in wavelength domain [W/(m^2 · sr · μm)]
    """
    mdict = load_mat_file_safe(rep_path)
    lam_ref, rad_um_ref = extract_lambda_and_radiance(mdict)
    return lam_ref, rad_um_ref


# (S6.5) Function — Title: Load all perturbation spectra and align to reference λ grid
def load_and_align_perturbations(
    data_root: Path,
    pattern: str,
    lam_ref: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads all 18 perturbation files and returns dictionary:
      out[key] with:
        - out[key]["pc"]: int (1..9)
        - out[key]["sign"]: str ("+" or "-")
        - out[key]["lam"]: wavelength μm (aligned to lam_ref; may be identical)
        - out[key]["rad_um"]: radiance in wavelength domain aligned to lam_ref
      where key is e.g., "PC1_plus", "PC1_minus", ... "PC9_minus"

    If a perturbation file's λ grid differs from lam_ref, it is interpolated onto lam_ref.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for idx in range(1, 19):
        fname = pattern.format(idx=idx)
        fpath = data_root / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Perturbation .mat not found: {fpath}")

        pc, sgn = index_to_pc_sign(idx)
        key = f"PC{pc}_{'plus' if sgn == '+' else 'minus'}"

        mdict = load_mat_file_safe(fpath)
        lam, rad_um = extract_lambda_and_radiance(mdict)

        # Align to reference λ grid if needed
        if not np.allclose(lam, lam_ref, rtol=0, atol=0):
            # Use linear interpolation; extrapolation should not be needed if grids overlap
            rad_interp = np.interp(lam_ref, lam, rad_um)
            lam_use = lam_ref.copy()
            rad_use = rad_interp
        else:
            lam_use = lam
            rad_use = rad_um

        out[key] = {
            "pc": pc,
            "sign": sgn,
            "lam": lam_use,
            "rad_um": rad_use
        }

    return out


# (S6.6) Function — Title: Compute ΔI spectra relative to representative reference
def compute_delta_I(
    rad_ref: np.ndarray,
    pert_dict: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """
    For each perturbation key, compute:
      ΔI(λ) = I_pert(λ) − I_ref(λ), retaining sign.
    Returns dict: key -> delta_I array aligned with lam_ref.
    """
    delta: Dict[str, np.ndarray] = {}
    for key, rec in pert_dict.items():
        rad_pert = rec["rad_um"]
        if rad_pert.shape != rad_ref.shape:
            raise ValueError(f"Shape mismatch for {key}: {rad_pert.shape} vs ref {rad_ref.shape}")
        delta[key] = rad_pert - rad_ref
    return delta


# (S6.7) Function — Title: Save CSVs (combined and per-PC) for ΔI(λ)
def save_deltaI_csvs(
    out_dir: Path,
    lam_ref: np.ndarray,
    deltaI: Dict[str, np.ndarray]
) -> Dict[str, str]:
    """
    Saves:
      - combined CSV: columns = all 18 ΔI(λ) series; index = λ_ref (μm)
      - per-PC CSVs: one per perturbation key, with two cols: [lambda_um, DeltaI]
    All CSVs include headers and row index (lambda_um).
    Returns mapping of logical names to paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_PC").mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}

    # Combined DataFrame
    df_comb = pd.DataFrame({k: v for k, v in deltaI.items()})
    df_comb.index = lam_ref
    df_comb.index.name = "lambda_um"
    f_comb = out_dir / "DeltaI_weighting_functions_all.csv"
    df_comb.to_csv(f_comb, index=True)
    files["DeltaI_weighting_functions_all.csv"] = str(f_comb)

    # Per-PC
    for key, arr in deltaI.items():
        df_pc = pd.DataFrame({"DeltaI_W_m2_sr_um": arr}, index=lam_ref)
        df_pc.index.name = "lambda_um"
        f_pc = out_dir / "per_PC" / f"{key}_weighting_function.csv"
        df_pc.to_csv(f_pc, index=True)
        files[f"per_PC/{key}_weighting_function.csv"] = str(f_pc)

    return files


# (S6.8) Function — Title: Make 18 per-perturbation plots of ΔI(λ)
def make_deltaI_plots(
    out_dir: Path,
    lam_ref: np.ndarray,
    deltaI: Dict[str, np.ndarray]
) -> List[str]:
    """
    Creates 18 figures, each ΔI(λ) vs λ (μm), retaining sign.
    Saves under out_dir/plots.
    """
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: List[str] = []

    # Sort keys in PC order for a predictable sequence
    def sort_key(k: str) -> Tuple[int, int]:
        # "PC{pc}_plus/minus"
        pc_str, sign_str = k.split("_")
        pc = int(pc_str.replace("PC", ""))
        sign_order = 0 if sign_str == "plus" else 1
        return (pc, sign_order)

    for key in sorted(deltaI.keys(), key=sort_key):
        y = deltaI[key]
        pc_str, sign_str = key.split("_")
        sign_disp = "+Δc" if sign_str == "plus" else "−Δc"

        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        ax.plot(lam_ref, y, color="crimson" if sign_str == "plus" else "royalblue", lw=1.5)
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.6)
        ax.set_xlabel("Wavelength λ (μm)")
        ax.set_ylabel("ΔI(λ) [W / (m² · sr · μm)]")
        ax.set_title(f"Weighting Function — {pc_str} ({sign_disp})")
        ax.grid(True, alpha=0.35)

        out_path = plots_dir / f"{key}_DeltaI_plot.png"
        fig.savefig(out_path)
        plt.close(fig)
        plot_paths.append(str(out_path))

    return plot_paths


# (S6.9) Function — Title: Write a detailed inventory log for Step 6
def write_inventory_log_step6(
    out_dir: Path,
    data_root: Path,
    rep_file: Path,
    files_map: Dict[str, str],
    plot_paths: List[str]
) -> str:
    """
    Writes a human-readable log summarizing:
      - Inputs (data paths)
      - Core relation: ΔI(λ) = I_pert(λ) − I_ref(λ)
      - Detailed outputs and meanings (CSVs and plots)
    """
    log_path = out_dir / "Step6_WeightingFunctions_inventory.log"

    lines: List[str] = []
    lines.append("Ground-based Atmospheric Profiles — Step 6: PCA-based Radiance Perturbation Spectra (Weighting Functions)")
    lines.append("")
    lines.append("Inputs used:")
    lines.append(f"  - Data root: {str(data_root.resolve())}")
    lines.append(f"  - Representative .mat: {str(rep_file.resolve())}")
    lines.append("  - Perturbation .mat files: Rad_DeltaC_TDp_index1.mat ... Rad_DeltaC_TDp_index18.mat")
    lines.append("")
    lines.append("Core relation (retain sign):")
    lines.append("  - ΔI_i,±(λ) = I_pert,i,±(λ) − I_ref(λ)")
    lines.append("    where:")
    lines.append("      λ: wavelength (μm)")
    lines.append("      I(λ): downwelling spectral radiance in wavelength domain [W/(m² · sr · μm)]")
    lines.append("      i ∈ {1..9}, ± denotes +Δc_i and −Δc_i")
    lines.append("")
    lines.append("Outputs saved in this step (all CSVs include headers and a row index named 'lambda_um'):")
    for k, v in files_map.items():
        if k == "DeltaI_weighting_functions_all.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: Combined table of ΔI(λ) for all 18 perturbations; columns are 'PCx_plus/minus'.")
        elif k.startswith("per_PC/"):
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: Single-perturbation ΔI(λ) series for the indicated PC and sign.")
        else:
            lines.append(f"  - {k}: {v}")
    lines.append("")
    if plot_paths:
        lines.append("Figures (per perturbation):")
        for p in plot_paths:
            lines.append(f"  - {p}")
        lines.append("  Each figure shows ΔI(λ) vs λ for one perturbation (PC i, +Δc_i or −Δc_i).")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - The sign of ΔI is preserved (do not take absolute value).")
    lines.append("  - No detector system effects (filters, ZnSe, LiTaO3 emission, response, gain, or NEP) are included.")
    lines.append("  - All outputs are written without timestamps for reproducibility.")
    lines.append("")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    return str(log_path)


# Main runner — Title: run_step6_weighting_functions_main (no CLI parsing)
def run_step6_weighting_functions_main():
    """
    Orchestrates Step 6:
      1) Load representative spectrum (λ_ref, I_ref(λ)).
      2) Load 18 perturbation spectra and align to λ_ref.
      3) Compute ΔI(λ) for each perturbation.
      4) Save combined and per-PC CSVs (with index and headers).
      5) Generate 18 ΔI(λ) plots.
      6) Write inventory log describing all outputs.
    """
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Representative spectrum
    rep_path = DATA_ROOT / REPRESENTATIVE_FILENAME
    if not rep_path.exists():
        raise FileNotFoundError(f"Representative .mat not found: {rep_path}")
    lam_ref, rad_um_ref = load_reference_spectrum(rep_path)

    # 2) Perturbations and alignment to λ_ref
    pert_dict = load_and_align_perturbations(DATA_ROOT, PERTURBATION_PATTERN, lam_ref)

    # 3) ΔI(λ)
    deltaI = compute_delta_I(rad_um_ref, pert_dict)

    # 4) Save CSVs
    files_map = save_deltaI_csvs(out_dir, lam_ref, deltaI)

    # 5) Plots
    plot_paths = make_deltaI_plots(out_dir, lam_ref, deltaI)

    # 6) Inventory log
    log_path = write_inventory_log_step6(out_dir, DATA_ROOT, rep_path, files_map, plot_paths)

    print("Step 6 completed.")
    print(f"- Outputs dir: {str(out_dir.resolve())}")
    print(f"- Inventory log: {log_path}")


# Entry point for VSCode "Run Python File"
if __name__ == "__main__":
    run_step6_weighting_functions_main()