# (J5) Step 5 (Joint) — Generate Perturbed Profiles from Joint PCA Modes
# Title: step5_joint_perturbation.py
# Author: (Your Lab)
# Description:
#   Uses joint PCA outputs (T + Td concatenated, z-score scaling) to generate
#   perturbed profiles about the representative profile (index=280), for k=9 PCs
#   (99% cumulative explained variance mode). Produces 9 figures (one per PC)
#   overlaying representative T/Td with ±Δc_i perturbations, and saves detailed CSVs.
#
# How to run (main program):
#   1) Edit USER PATHS below to point to your Step 1, Step 2 Joint, Step 3 Joint, Step 4 directories.
#   2) Run: run_step5_joint_main()
#   No command-line parsing is used. Outputs are saved under Step5_Perturbation_Joint (no timestamp).
#
# Key variables (readable, with meanings):
#   - H: height vector (21,) in meters (0–2 km levels), used for indexing and plotting.
#   - T_rep, Td_rep: representative temperature and dew-point profiles (length 21) from Step 4 (index=280).
#   - W_T (21 × k), W_Td (21 × k): joint PCA loadings split into T and Td blocks (columns PC1..PCk) from Step 2 Joint.
#     These are loadings in the "z-score scaled" feature space used by Step 2 Joint PCA.
#   - s_T (21,), s_Td (21,): per-feature standard deviations for T and Td computed from Step 1 centered data.
#     These are needed to convert perturbations from scaled space back to physical units.
#   - Δc_i (length k): recommended coefficient differentials from Step 3 Joint (delta_c_recommended.csv).
#   - Δx_scaled (T/Td): Δx_scaled_T = Δc_i · w_T_i, Δx_scaled_Td = Δc_i · w_Td_i (dimensionless in z-score space).
#   - Δx_physical: Δx_T = s_T ⊙ Δx_scaled_T, Δx_Td = s_Td ⊙ Δx_scaled_Td; used to perturb T_rep and Td_rep in physical units.
#
# Notes:
#   - Dew-point clipping (Td ≤ T) can be enabled via ENFORCE_TD_LE_T.
#   - We also annotate figures with PC explained variance ratio (EVR) if available.
#   - This step uses only the coefficient statistics and loadings; no detector system effects are involved.

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # for batch runs
import matplotlib.pyplot as plt

# =========================
# USER PATHS (edit here)
# =========================
STEP1_DIR = Path("./Step1_Processed")
# Example: Path to your Step 2 Joint PCA directory containing PCA_joint_W_T_k99.csv, PCA_joint_W_Td_k99.csv, etc.
STEP2_JOINT_DIR = Path(r"C:\Users\24573\OneDrive - The University of Hong Kong\PhD\Project\atmospheric sounding\prior-analysis\Step2_JointPCA")
# Path to your Step 3 Joint outputs containing Joint_k99/delta_c_recommended.csv (no timestamp by design)
STEP3_JOINT_DIR = Path("./Step3_Stats_Joint")
# Path to Step 4 representative profiles (no timestamp by design); contains Representative_T_profile_2km.csv, Representative_Td_profile_2km.csv
STEP4_DIR = Path("./Step4_RepresentativeProfile")

# Output root (no timestamp)
OUT_DIR = Path("./Step5_Perturbation_Joint")

# Options
ENFORCE_TD_LE_T = True    # enforce Td ≤ T after dew-point perturbations
K_MODE = "k99"            # use the 99% cumulative variance mode (k=9 PCs)

# Plot style
plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


# (J5.1) Function — Title: Load representative T/Td profiles and heights from Step 4
def load_representative_profiles_step4(step4_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - H (21,): heights
      - T_rep (21,): representative temperature (index=280) over 0–2 km
      - Td_rep (21,): representative dew-point (index=280) over 0–2 km
    """
    fT = step4_dir / "Representative_T_profile_2km.csv"
    fTd = step4_dir / "Representative_Td_profile_2km.csv"
    if not fT.exists() or not fTd.exists():
        raise FileNotFoundError("Step 4 representative profile CSVs not found.")

    dfT = pd.read_csv(fT, index_col=0)
    dfTd = pd.read_csv(fTd, index_col=0)
    required_T = {"H", "T_atm"}
    required_Td = {"H", "T_DP"}
    if not required_T.issubset(dfT.columns) or not required_Td.issubset(dfTd.columns):
        raise ValueError("Representative CSVs must contain columns: (H, T_atm) and (H, T_DP).")

    H = dfT["H"].to_numpy(dtype=float)
    T_rep = dfT["T_atm"].to_numpy(dtype=float)
    Td_rep = dfTd["T_DP"].to_numpy(dtype=float)
    return H, T_rep, Td_rep


# (J5.2) Function — Title: Load joint W (T/Td blocks) and Δc_i (k=9)
def load_joint_W_and_delta_k99(step2_joint_dir: Path,
                               step3_joint_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads:
      - W_T_k99: (21 × 9) with height index
      - W_Td_k99: (21 × 9) with height index
      - EVR_k99: explained variance ratios (PC1..PC9) if available (else NaNs)
      - delta_c (length 9) from Joint_k99/delta_c_recommended.csv
    """
    f_W_T = step2_joint_dir / "PCA_joint_W_T_k99.csv"
    f_W_Td = step2_joint_dir / "PCA_joint_W_Td_k99.csv"
    f_evr = step2_joint_dir / "PCA_joint_explained_variance_ratio.csv"
    f_delta = step3_joint_dir / "Joint_k99" / "delta_c_recommended.csv"

    if not f_W_T.exists() or not f_W_Td.exists():
        # fallback to full W and slice first 9 PCs
        f_W_T = step2_joint_dir / "PCA_joint_W_T.csv"
        f_W_Td = step2_joint_dir / "PCA_joint_W_Td.csv"
        if not f_W_T.exists() or not f_W_Td.exists():
            raise FileNotFoundError("Joint W_T(.csv) and/or W_Td(.csv) not found in Step 2 Joint directory.")

    W_T = pd.read_csv(f_W_T, index_col=0)
    W_Td = pd.read_csv(f_W_Td, index_col=0)

    # Keep first 9 PCs if more provided
    pcs_all = list(W_T.columns)
    pcs_k9 = [c for c in pcs_all if c.startswith("PC")][:9]
    W_T = W_T.loc[:, pcs_k9]
    W_Td = W_Td.loc[:, pcs_k9]

    # EVR (optional, annotate titles)
    if f_evr.exists():
        evr_df = pd.read_csv(f_evr, index_col=0)
        # attempt to normalize to a Series labeled 'explained_variance_ratio'
        if evr_df.shape[1] == 1:
            evr_full = evr_df.iloc[:, 0]
        else:
            if "explained_variance_ratio" in evr_df.columns:
                evr_full = evr_df["explained_variance_ratio"]
            else:
                evr_full = evr_df.iloc[:, 0]
        EVR_k9 = evr_full.iloc[:len(pcs_k9)].copy()
        EVR_k9.index = pcs_k9
    else:
        EVR_k9 = pd.Series([np.nan] * len(pcs_k9), index=pcs_k9, name="explained_variance_ratio")

    # Δc_i
    if not f_delta.exists():
        raise FileNotFoundError(f"delta_c_recommended.csv not found at: {f_delta}")
    delta_df = pd.read_csv(f_delta, index_col=0)
    # Expect either 'delta_c' column with PC index, or single-row with PC columns
    if "delta_c" in delta_df.columns:
        if "PC" in delta_df.columns:
            delta_s = delta_df.set_index("PC")["delta_c"].reindex(pcs_k9)
        else:
            delta_s = delta_df["delta_c"].reindex(pcs_k9)
    else:
        if all(col.startswith("PC") for col in delta_df.columns):
            delta_s = delta_df.iloc[0][pcs_k9]
        else:
            # Otherwise, take the last column as widths, indexed by PC labels
            delta_s = delta_df.iloc[:, -1]
            delta_s.index = pcs_k9[:len(delta_s)]
            delta_s = delta_s.reindex(pcs_k9)

    if delta_s.isna().any():
        raise ValueError("Δc_i could not be aligned to PC1..PC9 for joint analysis.")

    return W_T, W_Td, EVR_k9, delta_s


# (J5.3) Function — Title: Compute per-feature std from Step 1 centered matrices (z-score undo)
def compute_feature_stds_from_step1(step1_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads centered matrices to compute per-feature standard deviations (ddof=1):
      - s_T  (21,)
      - s_Td (21,)
    These undo z-score scaling: Δx_physical = s ⊙ Δx_scaled.
    """
    fXT = step1_dir / "X_T_centered.csv"
    fXTd = step1_dir / "X_Td_centered.csv"
    if not fXT.exists() or not fXTd.exists():
        raise FileNotFoundError("Step 1 centered matrices not found to derive per-feature std (s_T, s_Td).")

    XT = pd.read_csv(fXT, index_col=0).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    XTd = pd.read_csv(fXTd, index_col=0).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    s_T = np.std(XT, axis=0, ddof=1)
    s_Td = np.std(XTd, axis=0, ddof=1)
    # Avoid zeros
    s_T[s_T == 0] = 1.0
    s_Td[s_Td == 0] = 1.0
    return s_T, s_Td


# (J5.4) Function — Title: Construct physical perturbations for each PC (T and Td)
def construct_joint_perturbations_physical(
    W_T: pd.DataFrame,
    W_Td: pd.DataFrame,
    delta_s: pd.Series,
    s_T: np.ndarray,
    s_Td: np.ndarray,
    T_rep: np.ndarray,
    Td_rep: np.ndarray
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    For each PC i:
      - Δx_scaled_T  = Δc_i * w_T[:, i]
      - Δx_scaled_Td = Δc_i * w_Td[:, i]
      - Δx_T  = s_T  ⊙ Δx_scaled_T
      - Δx_Td = s_Td ⊙ Δx_scaled_Td
      - plus/minus: rep ± Δx (for both T and Td)

    Returns dict with DataFrames (indexed by W_T.index=H) for:
      - 'T': { 'delta_plus_theoretical', 'delta_minus_theoretical', 'plus', 'minus' }
      - 'Td': { 'delta_plus_theoretical', 'delta_minus_theoretical', 'plus', 'minus' }
    """
    pcs = list(W_T.columns)
    H_index = W_T.index

    # Build matrices (21 × k)
    dT_scaled = W_T.values * delta_s.values  # Δc_i * w_T
    dTd_scaled = W_Td.values * delta_s.values

    # Undo z-score scaling
    dT_phys = dT_scaled * s_T.reshape(-1, 1)
    dTd_phys = dTd_scaled * s_Td.reshape(-1, 1)

    # Plus/minus profiles
    T_plus = T_rep.reshape(-1, 1) + dT_phys
    T_minus = T_rep.reshape(-1, 1) - dT_phys
    Td_plus = Td_rep.reshape(-1, 1) + dTd_phys
    Td_minus = Td_rep.reshape(-1, 1) - dTd_phys

    # Package
    out = {
        "T": {
            "delta_plus_theoretical": pd.DataFrame(dT_phys, index=H_index, columns=pcs),
            "delta_minus_theoretical": pd.DataFrame(-dT_phys, index=H_index, columns=pcs),
            "plus": pd.DataFrame(T_plus, index=H_index, columns=pcs),
            "minus": pd.DataFrame(T_minus, index=H_index, columns=pcs),
        },
        "Td": {
            "delta_plus_theoretical": pd.DataFrame(dTd_phys, index=H_index, columns=pcs),
            "delta_minus_theoretical": pd.DataFrame(-dTd_phys, index=H_index, columns=pcs),
            "plus": pd.DataFrame(Td_plus, index=H_index, columns=pcs),
            "minus": pd.DataFrame(Td_minus, index=H_index, columns=pcs),
        }
    }
    return out


# (J5.5) Function — Title: Enforce dew-point ≤ temperature (optional)
def enforce_td_le_t(T_ref: pd.Series, Td_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of Td_df with each column clipped such that Td ≤ T_ref level-wise.
    T_ref is a Series indexed by height; Td_df is DataFrame indexed by height.
    """
    Td_clipped = Td_df.copy()
    # Align by index order; apply row-wise min
    for col in Td_clipped.columns:
        Td_clipped[col] = np.minimum(Td_clipped[col].values, T_ref.values)
    return Td_clipped


# (J5.6) Function — Title: Save all outputs (CSVs) for Joint_k99
def save_outputs_joint_step5(
    out_dir: Path,
    H: np.ndarray,
    T_rep: np.ndarray,
    Td_rep: np.ndarray,
    perturb: Dict[str, Dict[str, pd.DataFrame]],
    EVR_k9: pd.Series,
    delta_s: pd.Series,
    Td_plus_eff: pd.DataFrame,
    Td_minus_eff: pd.DataFrame
) -> Dict[str, str]:
    """
    Saves:
      - Representative profiles
      - W-derived deltas and plus/minus profiles (T, Td theoretical)
      - Effective Td deltas after clipping (computed as plus/minus - rep)
      - Per-PC CSVs with complete details
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_PC").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    files = {}
    # Representative
    df_rep = pd.DataFrame({"H": H, "T_rep": T_rep, "Td_rep": Td_rep})
    f_rep = out_dir / "Representative_T_Td_profile_2km.csv"
    df_rep.to_csv(f_rep, index=True)
    files["Representative_T_Td_profile_2km.csv"] = str(f_rep)

    # Bulk matrices
    for var in ["T", "Td"]:
        for key, df in perturb[var].items():
            path = out_dir / f"{var}_{key}.csv"
            df.to_csv(path, index=True)
            files[f"{var}_{key}.csv"] = str(path)

    # Effective Td deltas after clipping
    # Compute effective deltas relative to Td_rep
    Td_plus_eff_delta = Td_plus_eff.values - Td_rep.reshape(-1, 1)
    Td_minus_eff_delta = Td_minus_eff.values - Td_rep.reshape(-1, 1)
    df_Td_plus_eff_delta = pd.DataFrame(Td_plus_eff_delta, index=perturb["Td"]["plus"].index, columns=perturb["Td"]["plus"].columns)
    df_Td_minus_eff_delta = pd.DataFrame(Td_minus_eff_delta, index=perturb["Td"]["minus"].index, columns=perturb["Td"]["minus"].columns)

    f_Td_plus_eff = out_dir / "Td_delta_plus_effective.csv"
    f_Td_minus_eff = out_dir / "Td_delta_minus_effective.csv"
    df_Td_plus_eff_delta.to_csv(f_Td_plus_eff, index=True)
    df_Td_minus_eff_delta.to_csv(f_Td_minus_eff, index=True)
    files["Td_delta_plus_effective.csv"] = str(f_Td_plus_eff)
    files["Td_delta_minus_effective.csv"] = str(f_Td_minus_eff)

    # EVR_k9 and delta_c
    if EVR_k9 is not None and EVR_k9.size > 0:
        f_evr = out_dir / "explained_variance_ratio_k99.csv"
        EVR_k9.to_csv(f_evr, header=True)
        files["explained_variance_ratio_k99.csv"] = str(f_evr)

    df_delta = pd.DataFrame({"PC": delta_s.index, "delta_c": delta_s.values})
    f_delta = out_dir / "delta_c_used_k99.csv"
    df_delta.to_csv(f_delta, index=False)
    files["delta_c_used_k99.csv"] = str(f_delta)

    # Per-PC detailed CSVs
    pcs = list(delta_s.index)
    H_index = perturb["T"]["plus"].index
    for pc in pcs:
        df_pc = pd.DataFrame({
            "H": H_index.values,
            "T_rep": T_rep,
            "Td_rep": Td_rep,
            "T_plus": perturb["T"]["plus"][pc].values,
            "T_minus": perturb["T"]["minus"][pc].values,
            "Td_plus_theoretical": perturb["Td"]["plus"][pc].values,
            "Td_minus_theoretical": perturb["Td"]["minus"][pc].values,
            "Td_plus_effective": Td_plus_eff[pc].values,
            "Td_minus_effective": Td_minus_eff[pc].values,
            "delta_c": np.full_like(T_rep, delta_s[pc], dtype=float),
            "w_T_norm": np.linalg.norm(perturb["T"]["delta_plus_theoretical"][pc].values / delta_s[pc]) if delta_s[pc] != 0 else np.nan,
            "w_Td_norm": np.linalg.norm(perturb["Td"]["delta_plus_theoretical"][pc].values / delta_s[pc]) if delta_s[pc] != 0 else np.nan
        })
        f_pc = out_dir / "per_PC" / f"{pc}_joint_perturbation.csv"
        df_pc.to_csv(f_pc, index=False)
        files[f"per_PC/{pc}_joint_perturbation.csv"] = str(f_pc)

    return files


# (J5.7) Function — Title: Generate 9 per-PC figures overlaying rep and ±Δc_i perturbations
def make_per_pc_plots(out_dir: Path,
                      H: np.ndarray,
                      T_rep: np.ndarray,
                      Td_rep: np.ndarray,
                      perturb: Dict[str, Dict[str, pd.DataFrame]],
                      Td_plus_eff: pd.DataFrame,
                      Td_minus_eff: pd.DataFrame,
                      EVR_k9: pd.Series) -> List[str]:
    """
    For each PC (1..9), create a 2-panel figure:
      - Left: Temperature vs H (rep, plus, minus)
      - Right: Dew-point vs H (rep, plus_effective, minus_effective), with theoretical overlays
    Saves PNG files under out_dir/plots.
    """
    plot_paths = []
    pcs = list(perturb["T"]["plus"].columns)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for pc in pcs:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True, sharey=True)

        # Temperature
        ax = axes[0]
        ax.plot(T_rep, H, color="k", lw=1.8, label="T_rep (idx=280)")
        ax.plot(perturb["T"]["plus"][pc], H, color="crimson", lw=1.6, label=f"T_plus ({pc})")
        ax.plot(perturb["T"]["minus"][pc], H, color="royalblue", lw=1.6, label=f"T_minus ({pc})")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Height (m)")
        ax.set_title("Temperature perturbation")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", frameon=True)

        # Dew-point
        ax = axes[1]
        ax.plot(Td_rep, H, color="k", lw=1.8, label="Td_rep (idx=280)")
        # Effective (post-clipping)
        ax.plot(Td_plus_eff[pc], H, color="crimson", lw=1.6, label=f"Td_plus_eff ({pc})")
        ax.plot(Td_minus_eff[pc], H, color="royalblue", lw=1.6, label=f"Td_minus_eff ({pc})")
        # Optional: faint theoretical overlays
        ax.plot(perturb["Td"]["plus"][pc], H, color="crimson", lw=1.0, ls="--", alpha=0.6, label="Td_plus_theor")
        ax.plot(perturb["Td"]["minus"][pc], H, color="royalblue", lw=1.0, ls="--", alpha=0.6, label="Td_minus_theor")

        ax.set_xlabel("Dew-point")
        ax.set_title("Dew-point perturbation")
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", frameon=True)

        evr = float(EVR_k9.get(pc, np.nan)) if EVR_k9 is not None else np.nan
        fig.suptitle(f"Joint PCA — {pc}   (EVR≈{evr:.3f})", fontsize=13)

        out_path = plots_dir / f"{pc}_T_Td_perturbation.png"
        fig.savefig(out_path)
        plt.close(fig)
        plot_paths.append(str(out_path))

    return plot_paths


# (J5.8) Function — Title: Write a detailed inventory log for Step 5 (Joint)
def write_inventory_log_joint_step5(out_dir: Path,
                                    files_map: Dict[str, str],
                                    plot_paths: List[str],
                                    context: Dict[str, str]) -> str:
    log_path = out_dir / "Step5_Perturbation_Joint_inventory.log"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("Ground-based Atmospheric Profiles — Step 5 (Joint): Generate Perturbed Profiles")
    lines.append(f"Timestamp: {now}")
    lines.append("")
    lines.append("Inputs used:")
    lines.append(f"  - Step 1 (for s_T, s_Td): {context.get('step1_dir','')}")
    lines.append(f"  - Step 2 Joint (W_T_k99, W_Td_k99, EVR): {context.get('step2_joint_dir','')}")
    lines.append(f"  - Step 3 Joint (Δc_i): {context.get('step3_joint_dir','')}")
    lines.append(f"  - Step 4 (Representative idx=280): {context.get('step4_dir','')}")
    lines.append("")
    lines.append("Core relations (z-score scaled joint PCA, mapped back to physical units):")
    lines.append("  - Δx_scaled_T  = Δc_i · w_T_i,   Δx_scaled_Td = Δc_i · w_Td_i")
    lines.append("  - Δx_T  = s_T  ⊙ Δx_scaled_T,    Δx_Td       = s_Td ⊙ Δx_scaled_Td")
    lines.append("  - T_pert = T_rep ± Δx_T,         Td_pert     = Td_rep ± Δx_Td (with optional Td ≤ T clipping)")
    lines.append("")
    lines.append("Outputs saved in this step:")
    for k, v in files_map.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    if plot_paths:
        lines.append("Figures (per PC):")
        for p in plot_paths:
            lines.append(f"  - {p}")
        lines.append("  Each figure contains two panels: Temperature and Dew-point overlays (rep, +Δc_i, −Δc_i).")
    lines.append("")
    lines.append(f"Options: ENFORCE_TD_LE_T = {ENFORCE_TD_LE_T}")
    lines.append("Notes:")
    lines.append("  - Representative profile corresponds to sample index 280 (from Step 4).")
    lines.append("  - EVR annotations are included when available.")
    lines.append("  - No detector system effects are included in this step.")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    return str(log_path)


# Main runner — Title: run_step5_joint_main (no CLI parsing)
def run_step5_joint_main():
    # Prepare output folder
    out_dir = OUT_DIR / "Joint_k99"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Step 4 representative profiles
    H, T_rep, Td_rep = load_representative_profiles_step4(STEP4_DIR)

    # Load Step 2 joint W and Step 3 Δc_i (k=9)
    W_T, W_Td, EVR_k9, delta_s = load_joint_W_and_delta_k99(STEP2_JOINT_DIR, STEP3_JOINT_DIR)

    # Derive per-feature stds from Step 1 centered matrices (to undo z-score scaling)
    s_T, s_Td = compute_feature_stds_from_step1(STEP1_DIR)

    # Build physical perturbations
    perturb = construct_joint_perturbations_physical(W_T, W_Td, delta_s, s_T, s_Td, T_rep, Td_rep)

    # Enforce Td ≤ T if needed (effective Td after clipping)
    if ENFORCE_TD_LE_T:
        T_rep_series = pd.Series(T_rep, index=W_T.index, name="T_rep")
        Td_plus_eff = enforce_td_le_t(T_rep_series, perturb["Td"]["plus"])
        Td_minus_eff = enforce_td_le_t(T_rep_series, perturb["Td"]["minus"])
    else:
        Td_plus_eff = perturb["Td"]["plus"].copy()
        Td_minus_eff = perturb["Td"]["minus"].copy()

    # Save outputs
    files_map = save_outputs_joint_step5(
        out_dir=out_dir,
        H=H,
        T_rep=T_rep,
        Td_rep=Td_rep,
        perturb=perturb,
        EVR_k9=EVR_k9,
        delta_s=delta_s,
        Td_plus_eff=Td_plus_eff,
        Td_minus_eff=Td_minus_eff
    )

    # Make per-PC plots (9 figures)
    plot_paths = make_per_pc_plots(
        out_dir=out_dir,
        H=H,
        T_rep=T_rep,
        Td_rep=Td_rep,
        perturb=perturb,
        Td_plus_eff=Td_plus_eff,
        Td_minus_eff=Td_minus_eff,
        EVR_k9=EVR_k9
    )

    # Log
    context = {
        "step1_dir": str(STEP1_DIR.resolve()),
        "step2_joint_dir": str(STEP2_JOINT_DIR.resolve()),
        "step3_joint_dir": str(STEP3_JOINT_DIR.resolve()),
        "step4_dir": str(STEP4_DIR.resolve())
    }
    log_path = write_inventory_log_joint_step5(out_dir, files_map, plot_paths, context)

    print("Step 5 (Joint) completed.")
    print(f"- Outputs dir: {str(out_dir.resolve())}")
    print(f"- Inventory log: {log_path}")


# Entry point for VSCode "Run Python File"
if __name__ == "__main__":
    run_step5_joint_main()