# (S7) Step 7 — Information Content Analysis for Optimal Filter Placement
# Title: step7_information_content_analysis.py
# Author: (Your Lab)
# Description:
#   Implements Step 7 from the project proposal (README.md): Using the ΔI(λ) weighting
#   functions from Step 6 (18 perturbations: PC1..PC9, plus/minus), quantify spectral
#   "action regions" and inter-PC distinctiveness to guide optimal filter placement.
#   Outputs include:
#     - Overlay plots (all 18 ΔI and the 9 per-PC signed sensitivities)
#     - Sensitivity summary curves S_rms(λ), S_max(λ)
#     - Sliding-window metrics CSV (window-level integrated sensitivity per PC)
#     - Top-N windows by energy and per-PC dominance
#     - Pairwise PC correlation heatmap (based on signed sensitivity spectra)
#     - A detailed Step 7 inventory log describing all outputs and meanings
#
# Notes:
#   - Inputs from Step 6: ./Step6_PCA_WeightingFunctions/DeltaI_weighting_functions_all.csv
#       Index: 'lambda_um' (μm)
#       Columns: 'PC{1..9}_{plus|minus}'
#   - Units:
#       ΔI(λ): W / (m^2 · sr · μm)
#       Integrated window vector (per PC): ∫ W_i(λ) dλ [W / (m^2 · sr)]
#         where W_i(λ) := (ΔI_i,plus(λ) − ΔI_i,minus(λ)) / 2  (signed sensitivity per PC, up to a scale factor Δc_i)
#   - No detector system effects (filters, ZnSe, LiTaO3 emission, response, gain, NEP) in Step 7.
#   - No command-line parsing; outputs go to fixed directory without timestamps.
#
# How to run (main program):
#   1) Ensure Step 6 outputs exist at USER PATHS below.
#   2) Run: run_step7_information_content_main()
#
# Key variables (readable, with meanings):
#   - λ (lam): wavelength grid [μm], index read from the Step 6 combined CSV (Symbol: lam).
#   - ΔI_i,±(λ) (deltaI): weighting functions for each PC i ∈ {1..9}, ± ∈ {plus, minus}
#       Units: W/(m^2 · sr · μm). Dictionary of 18 series.
#   - W_i(λ) (pc_sens): per-PC signed sensitivity derived from half-difference:
#       W_i(λ) = 0.5 * (ΔI_i,plus(λ) − ΔI_i,minus(λ)), i ∈ {1..9}
#       Units: W/(m^2 · sr · μm) up to an unknown Δc_i scale; sufficient for relative analysis.
#   - S_rms(λ): RMS magnitude across spectra (either 18 ΔI or 9 W_i) at each λ, revealing "action regions".
#   - Sliding window [λ_c − Δλ/2, λ_c + Δλ/2]:
#       Integrated vector V(window) ∈ R^9 with components:
#         V_i = ∫window W_i(λ) dλ  (trapezoid rule, Units: W/(m^2 · sr))
#       Energy E = ||V||_2, Dominance Index DI = max_i |V_i| / Σ_i |V_i|, Dominant PC = argmax_i |V_i|.
#       Opposition Fraction: fraction of PCs whose sign(V_i) opposes sign of dominant component and |V_i| exceeds a small threshold.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# =========================
# USER PATHS (edit here)
# =========================
# Step 6 combined CSV (must exist)
STEP6_DIR = Path("./Step6_PCA_WeightingFunctions")
STEP6_COMBINED_CSV = STEP6_DIR / "DeltaI_weighting_functions_all.csv"

# Step 7 output root (no timestamp)
OUT_DIR = Path("./Step7_InfoContent_Analysis")

# Sliding-window defaults (μm)
WINDOW_WIDTH_UM_DEFAULT = 0.5
WINDOW_STRIDE_UM_DEFAULT = 0.1

# Number of top windows to report
TOP_N_BY_ENERGY = 12  # overall top windows by energy
TOP_1_PER_PC = True   # if True, also report top per-PC dominant windows

# Plot style
plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


# (S7.1) Function — Title: Load Step 6 combined ΔI(λ) CSV
def load_step6_deltaI_combined(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Loads the combined ΔI(λ) CSV produced by Step 6, with:
      - Index: λ (mu m), named 'lambda_um'
      - Columns: 'PC{1..9}_plus' and 'PC{1..9}_minus'
    Returns:
      - lam: np.ndarray shape [Nλ,], wavelength grid in micrometers (μm), ascending
      - deltaI: dict mapping column key -> np.ndarray [Nλ,] (W/(m^2 · sr · μm))
    """
    if not path.exists():
        raise FileNotFoundError(f"Step 6 combined CSV not found: {path}")
    df = pd.read_csv(path, index_col=0)
    # Ensure index name and convert to float
    lam = df.index.values.astype(float)
    # Ensure ascending λ
    if lam[0] > lam[-1]:
        df = df.iloc[::-1].copy()
        lam = df.index.values.astype(float)
    # Keep only PC columns (robust to extra columns)
    cols = [c for c in df.columns if c.startswith("PC") and ("plus" in c or "minus" in c)]
    df = df[cols].copy()
    deltaI = {c: df[c].values.astype(float) for c in df.columns}
    return lam, deltaI


# (S7.2) Function — Title: Build per-PC signed sensitivity W_i(λ) via half-difference
def build_pc_signed_sensitivity(
    lam: np.ndarray,
    deltaI: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Constructs per-PC signed sensitivity spectra:
      W_i(λ) = 0.5 * (ΔI_i,plus(λ) − ΔI_i,minus(λ)), i ∈ {1..9}
    Returns:
      - lam (pass-through)
      - pc_sens: dict with keys 'PC{i}_sens' -> np.ndarray [Nλ,]
    """
    # Collect plus/minus pairs
    pc_sens: Dict[str, np.ndarray] = {}
    for i in range(1, 10):
        k_plus = f"PC{i}_plus"
        k_minus = f"PC{i}_minus"
        if k_plus not in deltaI or k_minus not in deltaI:
            raise KeyError(f"Missing required columns for PC{i}: {k_plus} or {k_minus}")
        # Half-difference emphasizes the derivative direction; Δc_i scale not needed for relative analysis
        wi = 0.5 * (deltaI[k_plus] - deltaI[k_minus])
        pc_sens[f"PC{i}_sens"] = wi
    return lam, pc_sens


# (S7.3) Function — Title: Sensitivity summary curves S_rms(λ) and S_max(λ)
def compute_sensitivity_curves(
    fields: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a dict of spectral fields f_k(λ), compute:
      - S_rms(λ) = sqrt( mean_k( f_k(λ)^2 ) )
      - S_max(λ) = max_k( |f_k(λ)| )
    Returns:
      (S_rms, S_max) arrays, same length as λ.
    """
    # Stack into [Nλ, K]
    arr = np.vstack([v for v in fields.values()]).T
    S_rms = np.sqrt(np.mean(arr**2, axis=1))
    S_max = np.max(np.abs(arr), axis=1)
    return S_rms, S_max


# (S7.4) Function — Title: Sliding-window integration of per-PC sensitivity
def sliding_window_integrate_pc_sensitivity(
    lam: np.ndarray,
    pc_sens: Dict[str, np.ndarray],
    window_width_um: float,
    window_stride_um: float,
    min_points: int = 5
) -> pd.DataFrame:
    """
    For a sliding window [λc − Δλ/2, λc + Δλ/2], integrate per-PC sensitivity:
      V_i(window) = ∫ W_i(λ) dλ   (trapezoidal)
    Also compute:
      - Energy E = ||V||_2
      - Dominant PC (by |V_i|)
      - Dominance Index DI = max_i |V_i| / Σ_i |V_i|
      - Opposition Fraction (fraction of PCs whose sign opposes dominant sign and |V_i| > small threshold)
    Returns a DataFrame with one row per window:
      index: window_id
      cols: ['lambda_center_um','lambda_start_um','lambda_end_um','num_points','E_L2',
             'dominant_pc','dominant_sign','dominance_index','opposition_fraction',
             'V_PC1',...,'V_PC9']
    """
    lam = lam.astype(float)
    lam_min, lam_max = lam.min(), lam.max()
    centers = np.arange(lam_min + 0.5 * window_width_um, lam_max - 0.5 * window_width_um + 1e-12, window_stride_um)

    # Stack W_i(λ) as [Nλ, 9]
    pc_keys = [f"PC{i}_sens" for i in range(1, 10)]
    W = np.vstack([pc_sens[k] for k in pc_keys]).T  # [Nλ, 9]

    rows = []
    window_id = 0
    for c in centers:
        a = c - 0.5 * window_width_um
        b = c + 0.5 * window_width_um
        idx = np.where((lam >= a) & (lam <= b))[0]
        if idx.size < min_points:
            continue
        lam_w = lam[idx]
        W_w = W[idx, :]  # [n,9]

        # Integrate each PC curve over window (trapezoid)
        V = np.trapz(W_w, lam_w, axis=0)  # shape [9,], Units: W/(m^2 · sr)
        absV = np.abs(V)
        sum_absV = np.sum(absV)
        E = float(np.linalg.norm(V, ord=2))
        if sum_absV <= 0:
            # Degenerate window (all zeros)
            dominant_pc_idx = -1
            dom_sign = 0
            DI = 0.0
            opp_frac = 0.0
        else:
            dominant_pc_idx = int(np.argmax(absV))  # 0-based
            dom_sign = int(np.sign(V[dominant_pc_idx]))  # -1,0,+1
            DI = float(absV[dominant_pc_idx] / sum_absV)

            # Opposition Fraction: count PCs with opposite sign and significant magnitude
            thr = 0.01 * absV[dominant_pc_idx]  # 1% of dominant magnitude
            significant = absV >= thr
            signs = np.sign(V)
            opposite = (signs == -dom_sign) & significant
            # Exclude the dominant PC itself
            mask_count = np.ones_like(opposite, dtype=bool)
            mask_count[dominant_pc_idx] = False
            denom = max(1, np.sum(significant & mask_count))
            opp_frac = float(np.sum(opposite & mask_count) / denom)

        row = {
            "lambda_center_um": float(c),
            "lambda_start_um": float(a),
            "lambda_end_um": float(b),
            "num_points": int(idx.size),
            "E_L2": E,
            "dominant_pc": f"PC{dominant_pc_idx+1}" if dominant_pc_idx >= 0 else "None",
            "dominant_sign": int(dom_sign),
            "dominance_index": DI,
            "opposition_fraction": opp_frac,
        }
        for i in range(9):
            row[f"V_PC{i+1}"] = float(V[i])
        rows.append(row)
        window_id += 1

    df = pd.DataFrame(rows)
    df.index.name = "window_id"
    return df


# (S7.5) Function — Title: Non-maximum suppression (NMS) for selecting non-overlapping top windows
def select_top_windows_nonoverlap(
    df_windows: pd.DataFrame,
    score_col: str,
    top_n: int,
    min_separation_um: float
) -> pd.DataFrame:
    """
    Selects up to top_n windows with the highest score (score_col), enforcing
    a minimum separation in center wavelength to avoid redundant overlaps.
    Returns a filtered DataFrame (preserving original index and sorted by score desc).
    """
    df = df_windows.sort_values(score_col, ascending=False).copy()
    selected_idx = []
    selected_centers = []

    for idx, row in df.iterrows():
        c = row["lambda_center_um"]
        if all(abs(c - sc) >= min_separation_um for sc in selected_centers):
            selected_idx.append(idx)
            selected_centers.append(c)
        if len(selected_idx) >= top_n:
            break
    return df.loc[selected_idx]


# (S7.6) Function — Title: Pairwise correlation among PCs (full-band signed sensitivity)
def compute_pc_pairwise_correlation(pc_sens: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Computes Pearson correlation among the 9 signed sensitivity spectra W_i(λ) across the full band.
    Returns a 9x9 DataFrame with index/columns 'PC{i}_sens'.
    """
    keys = [f"PC{i}_sens" for i in range(1, 10)]
    mat = np.vstack([pc_sens[k] for k in keys])
    # Correlation matrix across curves (rows -> PCs)
    C = np.corrcoef(mat)
    return pd.DataFrame(C, index=keys, columns=keys)


# (S7.7) Function — Title: Plot overlays and curves
def make_plots_step7(
    out_dir: Path,
    lam: np.ndarray,
    deltaI: Dict[str, np.ndarray],
    pc_sens: Dict[str, np.ndarray],
    S_rms_18: np.ndarray,
    S_max_18: np.ndarray,
    S_rms_9: np.ndarray,
    S_max_9: np.ndarray,
    df_windows: pd.DataFrame,
    df_top_energy: pd.DataFrame,
    corr_df: pd.DataFrame
) -> List[str]:
    """
    Generates:
      - Overlay of all 18 ΔI(λ)
      - Overlay of 9 per-PC signed sensitivities W_i(λ)
      - Sensitivity curves S_rms/S_max (18 and 9 versions)
      - Pairwise PC correlation heatmap
      - Overlay with top energy windows shaded
    Returns list of plot file paths.
    """
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[str] = []

    # Overlay 18 ΔI
    fig, ax = plt.subplots(figsize=(10, 5))
    # Sort keys for reproducible color mapping
    def sort_key(k: str) -> Tuple[int, int]:
        pc = int(k.split("_")[0].replace("PC", ""))
        sgn = 0 if "plus" in k else 1
        return (pc, sgn)
    for key in sorted(deltaI.keys(), key=sort_key):
        arr = deltaI[key]
        color = "crimson" if "plus" in key else "royalblue"
        ax.plot(lam, arr, lw=1.0, alpha=0.9, label=key, color=color)
    ax.axhline(0.0, color="k", lw=1.0, alpha=0.6)
    ax.set_xlabel("Wavelength λ (μm)")
    ax.set_ylabel("ΔI(λ) [W / (m² · sr · μm)]")
    ax.set_title("Overlay — 18 Weighting Functions ΔI(λ)")
    # Use a single legend outside if too many
    ax.legend(ncol=3, fontsize=7, frameon=True, loc="upper right")
    p1 = plots_dir / "overlay_all_18_DeltaI.png"
    fig.tight_layout()
    fig.savefig(p1); plt.close(fig)
    out_paths.append(str(p1))

    # Overlay 9 signed sensitivities W_i
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    for i in range(1, 10):
        key = f"PC{i}_sens"
        ax.plot(lam, pc_sens[key], lw=1.4, alpha=0.95, label=key, color=cmap((i-1) % 10))
    ax.axhline(0.0, color="k", lw=1.0, alpha=0.6)
    ax.set_xlabel("Wavelength λ (μm)")
    ax.set_ylabel("W_i(λ) [W / (m² · sr · μm)]")
    ax.set_title("Overlay — 9 Per-PC Signed Sensitivities W_i(λ) = 0.5·(ΔI_plus − ΔI_minus)")
    ax.legend(ncol=3, fontsize=8, frameon=True, loc="upper right")
    p2 = plots_dir / "overlay_9_PC_signed_sensitivities.png"
    fig.tight_layout()
    fig.savefig(p2); plt.close(fig)
    out_paths.append(str(p2))

    # Sensitivity curves (18 and 9)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lam, S_rms_18, color="firebrick", lw=1.8, label="S_rms across 18 ΔI")
    ax.plot(lam, S_max_18, color="darkred", lw=1.0, alpha=0.8, label="S_max across 18 ΔI")
    ax.plot(lam, S_rms_9, color="royalblue", lw=1.8, label="S_rms across 9 W_i")
    ax.plot(lam, S_max_9, color="midnightblue", lw=1.0, alpha=0.8, label="S_max across 9 W_i")
    ax.set_xlabel("Wavelength λ (μm)")
    ax.set_ylabel("Sensitivity magnitude [W / (m² · sr · μm)]")
    ax.set_title("Sensitivity Curves — RMS and Max Magnitude vs λ")
    ax.legend(frameon=True)
    p3 = plots_dir / "sensitivity_curves_Srms_Smax.png"
    fig.tight_layout()
    fig.savefig(p3); plt.close(fig)
    out_paths.append(str(p3))

    # Pairwise correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(9)); ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(9)); ax.set_yticklabels(corr_df.index)
    ax.set_title("Pairwise Correlation among W_i(λ) across full band")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    p4 = plots_dir / "pairwise_PC_correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(p4); plt.close(fig)
    out_paths.append(str(p4))

    # Overlay of 9 W_i with top energy windows shaded
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(1, 10):
        key = f"PC{i}_sens"
        ax.plot(lam, pc_sens[key], lw=1.0, alpha=0.95, label=key)
    ymin = min(np.min(v) for v in pc_sens.values())
    ymax = max(np.max(v) for v in pc_sens.values())
    # Shade top energy windows
    for idx, row in df_top_energy.iterrows():
        a, b = row["lambda_start_um"], row["lambda_end_um"]
        ax.axvspan(a, b, color="gold", alpha=0.25)
    ax.axhline(0.0, color="k", lw=1.0, alpha=0.6)
    ax.set_ylim(ymin*1.05 if ymin<0 else ymin*0.95, ymax*1.05)
    ax.set_xlabel("Wavelength λ (μm)")
    ax.set_ylabel("W_i(λ) [W / (m² · sr · μm)]")
    ax.set_title("Top Energy Windows (shaded) over W_i(λ)")
    ax.legend(ncol=3, fontsize=8, frameon=True, loc="upper right")
    p5 = plots_dir / "overlay_Wi_with_top_energy_windows.png"
    fig.tight_layout()
    fig.savefig(p5); plt.close(fig)
    out_paths.append(str(p5))

    return out_paths


# (S7.8) Function — Title: Save CSVs (curves and windows) with headers and index
def save_step7_csvs(
    out_dir: Path,
    lam: np.ndarray,
    S_curves_18: Tuple[np.ndarray, np.ndarray],
    S_curves_9: Tuple[np.ndarray, np.ndarray],
    pc_sens: Dict[str, np.ndarray],
    df_windows: pd.DataFrame,
    df_top_energy: pd.DataFrame,
    df_top_per_pc: pd.DataFrame,
    corr_df: pd.DataFrame
) -> Dict[str, str]:
    """
    Saves:
      - S_curves_18.csv: λ-indexed S_rms and S_max across 18 ΔI
      - S_curves_9.csv:  λ-indexed S_rms and S_max across 9 W_i
      - PC_signed_sensitivity.csv: λ-indexed per-PC W_i(λ)
      - window_metrics.csv: window-level metrics (one row per window)
      - top_windows_by_energy.csv
      - top_windows_per_pc.csv (if TOP_1_PER_PC True; else empty)
      - PC_pairwise_correlation.csv: 9x9 correlation among W_i
    Returns map of logical names -> file path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}

    # Sensitivity curves (18 and 9)
    lam_series = pd.Index(lam, name="lambda_um")
    S18 = pd.DataFrame(
        {
            "S_rms_18": S_curves_18[0],
            "S_max_18": S_curves_18[1],
        },
        index=lam_series
    )
    f_S18 = out_dir / "S_curves_18.csv"
    S18.to_csv(f_S18, index=True)
    files["S_curves_18.csv"] = str(f_S18)

    S9 = pd.DataFrame(
        {
            "S_rms_9": S_curves_9[0],
            "S_max_9": S_curves_9[1],
        },
        index=lam_series
    )
    f_S9 = out_dir / "S_curves_9.csv"
    S9.to_csv(f_S9, index=True)
    files["S_curves_9.csv"] = str(f_S9)

    # PC signed sensitivity W_i(λ)
    df_W = pd.DataFrame({k: v for k, v in pc_sens.items()}, index=lam_series)
    f_W = out_dir / "PC_signed_sensitivity.csv"
    df_W.to_csv(f_W, index=True)
    files["PC_signed_sensitivity.csv"] = str(f_W)

    # Window metrics
    f_win = out_dir / "window_metrics.csv"
    df_windows.to_csv(f_win, index=True)
    files["window_metrics.csv"] = str(f_win)

    # Top windows by energy
    f_topE = out_dir / "top_windows_by_energy.csv"
    df_top_energy.to_csv(f_topE, index=True)
    files["top_windows_by_energy.csv"] = str(f_topE)

    # Top per-PC dominance (optional)
    if df_top_per_pc is not None and not df_top_per_pc.empty:
        f_topPC = out_dir / "top_windows_per_pc.csv"
        df_top_per_pc.to_csv(f_topPC, index=True)
        files["top_windows_per_pc.csv"] = str(f_topPC)

    # Pairwise correlation (9x9)
    f_corr = out_dir / "PC_pairwise_correlation.csv"
    corr_df.to_csv(f_corr, index=True)
    files["PC_pairwise_correlation.csv"] = str(f_corr)

    return files


# (S7.9) Function — Title: Write a detailed inventory log for Step 7
def write_inventory_log_step7(
    out_dir: Path,
    step6_csv: Path,
    window_width_um: float,
    window_stride_um: float,
    files_map: Dict[str, str],
    plot_paths: List[str]
) -> str:
    """
    Writes a human-readable log summarizing:
      - Inputs (Step 6 combined CSV)
      - Core definitions used in Step 7
      - Detailed outputs and meanings (CSVs and plots)
    """
    log_path = out_dir / "Step7_InfoContent_inventory.log"

    lines: List[str] = []
    lines.append("Ground-based Atmospheric Profiles — Step 7: Information Content Analysis for Optimal Filter Placement")
    lines.append("")
    lines.append("Inputs used:")
    lines.append(f"  - Step 6 combined CSV (ΔI): {str(step6_csv.resolve())}")
    lines.append("  - Note: ΔI(λ) units are W/(m² · sr · μm); sign is preserved from Step 6.")
    lines.append("")
    lines.append("Core definitions (retain sign):")
    lines.append("  - Per-PC signed sensitivity: W_i(λ) = 0.5 · (ΔI_i,plus(λ) − ΔI_i,minus(λ)), i ∈ {1..9}.")
    lines.append("  - Sensitivity curves: ")
    lines.append("      S_rms(λ) = sqrt( mean_k( f_k(λ)^2 ) ), S_max(λ) = max_k(|f_k(λ)|).")
    lines.append("    We provide both across 18 ΔI(λ) and across 9 W_i(λ).")
    lines.append("  - Sliding window (width Δλ, stride): For each window [λ_c − Δλ/2, λ_c + Δλ/2],")
    lines.append("      V_i = ∫ W_i(λ) dλ  (trapezoid), E = ||V||_2,")
    lines.append("      Dominance Index DI = max_i |V_i| / Σ_i |V_i|,")
    lines.append("      Opposition Fraction = fraction of PCs opposing the dominant sign with significant magnitude (≥1% of dominant).")
    lines.append(f"  - Window settings: width Δλ = {window_width_um:.3f} μm, stride = {window_stride_um:.3f} μm.")
    lines.append("")
    lines.append("Outputs (all CSVs include headers; λ-indexed files use 'lambda_um'):")
    for k, v in files_map.items():
        if k == "S_curves_18.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: λ-indexed S_rms and S_max computed across the 18 ΔI(λ) spectra.")
        elif k == "S_curves_9.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: λ-indexed S_rms and S_max computed across the 9 per-PC signed sensitivities W_i(λ).")
        elif k == "PC_signed_sensitivity.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: λ-indexed table of W_i(λ) = 0.5·(ΔI_plus − ΔI_minus) for i=1..9; sign retained.")
        elif k == "window_metrics.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: Sliding-window metrics. Columns include window bounds, number of points,")
            lines.append("             E_L2 (overall energy), dominant_pc, dominant_sign, dominance_index,")
            lines.append("             opposition_fraction, and integrated components V_PC1..V_PC9.")
        elif k == "top_windows_by_energy.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: Top-N windows ranked by E_L2 with non-overlap enforced; candidates for high-signal filter placement.")
        elif k == "top_windows_per_pc.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: For each PC, the most dominant high-energy window (if available); candidates to uniquely sense each PC.")
        elif k == "PC_pairwise_correlation.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: 9×9 Pearson correlation among W_i(λ) over the full band; lower |r| suggests better separability.")
        else:
            lines.append(f"  - {k}: {v}")
    lines.append("")
    if plot_paths:
        lines.append("Figures:")
        for p in plot_paths:
            lines.append(f"  - {p}")
        lines.append("  Meanings:")
        lines.append("    - overlay_all_18_DeltaI.png: All 18 ΔI(λ) overlaid (visual map of sensitivity).")
        lines.append("    - overlay_9_PC_signed_sensitivities.png: 9 derived per-PC signed sensitivities W_i(λ).")
        lines.append("    - sensitivity_curves_Srms_Smax.png: S_rms and S_max vs λ (18 ΔI and 9 W_i).")
        lines.append("    - pairwise_PC_correlation_heatmap.png: Inter-PC correlation (full-band).")
        lines.append("    - overlay_Wi_with_top_energy_windows.png: W_i(λ) overlaid with top energy windows shaded (candidate bands).")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - No detector system effects (filters, ZnSe, LiTaO3 emission, response, gain, or NEP) are included in Step 7.")
    lines.append("  - Outputs are written without timestamps for reproducibility.")
    lines.append("")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    return str(log_path)


# Main runner — Title: run_step7_information_content_main (no CLI parsing)
def run_step7_information_content_main():
    """
    Orchestrates Step 7:
      1) Load Step 6 combined ΔI(λ).
      2) Build 9 per-PC signed sensitivities W_i(λ) = 0.5·(ΔI_plus − ΔI_minus).
      3) Compute sensitivity curves S_rms and S_max across 18 ΔI and across 9 W_i.
      4) Sliding-window integration to derive window-level metrics (E_L2, dominance, opposition).
      5) Select top-N non-overlapping windows by energy; optionally top-1 per PC by dominance.
      6) Pairwise correlation among PCs (full-band).
      7) Save CSVs and plots.
      8) Write inventory log documenting all outputs and meanings.
    """
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load Step 6 ΔI(λ)
    lam, deltaI = load_step6_deltaI_combined(STEP6_COMBINED_CSV)

    # 2) Build per-PC signed sensitivities
    lam, pc_sens = build_pc_signed_sensitivity(lam, deltaI)

    # 3) Sensitivity curves (across 18 ΔI and across 9 W_i)
    S_rms_18, S_max_18 = compute_sensitivity_curves(deltaI)
    S_rms_9, S_max_9 = compute_sensitivity_curves(pc_sens)

    # 4) Sliding-window metrics
    df_windows = sliding_window_integrate_pc_sensitivity(
        lam=lam,
        pc_sens=pc_sens,
        window_width_um=WINDOW_WIDTH_UM_DEFAULT,
        window_stride_um=WINDOW_STRIDE_UM_DEFAULT,
        min_points=5
    )

    # 5) Top windows by energy (non-overlapping)
    df_top_energy = select_top_windows_nonoverlap(
        df_windows=df_windows,
        score_col="E_L2",
        top_n=TOP_N_BY_ENERGY,
        min_separation_um=0.5 * WINDOW_WIDTH_UM_DEFAULT
    )

    # Optional: Top-1 per PC by dominance among windows with high energy
    df_top_per_pc = pd.DataFrame()
    if TOP_1_PER_PC and not df_windows.empty:
        rows = []
        # Filter to higher-energy half to avoid trivial windows
        energy_thr = np.percentile(df_windows["E_L2"].values, 50.0)
        df_filt = df_windows[df_windows["E_L2"] >= energy_thr].copy()
        for i in range(1, 10):
            pc_name = f"PC{i}"
            dfi = df_filt[df_filt["dominant_pc"] == pc_name].copy()
            if dfi.empty:
                continue
            # Score by dominance_index * E_L2
            dfi["score_dom_energy"] = dfi["dominance_index"] * dfi["E_L2"]
            best = dfi.sort_values("score_dom_energy", ascending=False).head(1)
            rows.append(best)
        if rows:
            df_top_per_pc = pd.concat(rows, axis=0)
            df_top_per_pc = df_top_per_pc.sort_values("E_L2", ascending=False)

    # 6) Pairwise correlation among W_i (full band)
    corr_df = compute_pc_pairwise_correlation(pc_sens)

    # 7) Plots
    plot_paths = make_plots_step7(
        out_dir=out_dir,
        lam=lam,
        deltaI=deltaI,
        pc_sens=pc_sens,
        S_rms_18=S_rms_18,
        S_max_18=S_max_18,
        S_rms_9=S_rms_9,
        S_max_9=S_max_9,
        df_windows=df_windows,
        df_top_energy=df_top_energy,
        corr_df=corr_df
    )

    # 8) Save CSVs
    files_map = save_step7_csvs(
        out_dir=out_dir,
        lam=lam,
        S_curves_18=(S_rms_18, S_max_18),
        S_curves_9=(S_rms_9, S_max_9),
        pc_sens=pc_sens,
        df_windows=df_windows,
        df_top_energy=df_top_energy,
        df_top_per_pc=df_top_per_pc,
        corr_df=corr_df
    )

    # 9) Inventory log
    log_path = write_inventory_log_step7(
        out_dir=out_dir,
        step6_csv=STEP6_COMBINED_CSV,
        window_width_um=WINDOW_WIDTH_UM_DEFAULT,
        window_stride_um=WINDOW_STRIDE_UM_DEFAULT,
        files_map=files_map,
        plot_paths=plot_paths
    )

    print("Step 7 completed.")
    print(f"- Outputs dir: {str(out_dir.resolve())}")
    print(f"- Inventory log: {log_path}")


# Entry point for VSCode "Run Python File"
if __name__ == "__main__":
    run_step7_information_content_main()