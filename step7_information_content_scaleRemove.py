# (S7) Step 7 — Information Content Analysis for Optimal Filter Placement (Δc-normalized, 25 top energy windows, top-5 per PC)
# Title: step7_information_content_analysis_normalized.py
# Author: (Your Lab)
# Description:
#   Implements Step 7 with Δc normalization to remove bin-width bias:
#     W_i(λ) = 0.5 · (ΔI_i,plus − ΔI_i,minus) ≈ Δc_i · ∂I/∂c_i
#     U_i(λ) = W_i(λ) / Δc_i ≈ ∂I/∂c_i
#   New requirements:
#     - Select Top-25 non-overlapping windows by energy (U-based).
#     - For each PC, select Top-5 windows by per-PC score without requiring dominance:
#         score_i(window) = E_L2(window) × DI_i(window),
#         with DI_i(window) = |V_i| / Σ_j |V_j|, V_i(window) = ∫ U_i(λ) dλ
#     - Plot top windows for both categories (energy-only, and combined energy + per-PC).
#
# Notes:
#   - Inputs:
#       Step 6 combined CSV: ./Step6_PCA_WeightingFunctions/DeltaI_weighting_functions_all.csv
#       Step 5 Δc CSV:       ./Step5_Perturbation_Joint/Joint_k99/delta_c_used_k99.csv
#   - Units:
#       ΔI(λ), W_i(λ), U_i(λ): W/(m^2 · sr · μm)
#       V_i(window) = ∫ (·) dλ: W/(m^2 · sr)
#       E_L2: W/(m^2 · sr); dominance_index, opposition_fraction, DI_i: unitless.
#   - Outputs: ./Step7_InfoContent_Analysis (no timestamp), CSVs with headers and row index.
#
# How to run (main program):
#   1) Ensure Step 6 and Step 5 files exist at USER PATHS below.
#   2) Run: run_step7_information_content_main()
#
# Key variables (first-appearance meanings):
#   - λ (lam) [μm]: wavelength grid.
#   - ΔI_i,±(λ): Step 6 radiance-difference spectra per PC/sign [W/(m^2 · sr · μm)].
#   - Δc_i: coefficient perturbation magnitudes used in Step 5 (provided).
#   - W_i(λ) = 0.5 · (ΔI_i,plus − ΔI_i,minus) ≈ Δc_i · ∂I/∂c_i.
#   - U_i(λ) = W_i(λ) / Δc_i ≈ ∂I(λ)/∂c_i (unit-coefficient sensitivity).
#   - Sliding-window [λ_c − Δλ/2, λ_c + Δλ/2]: integrates W_i or U_i to V_i(window).
#   - E_L2, dominance_index, opposition_fraction: window metrics for ranking bands.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =========================
# USER PATHS (edit here)
# =========================
STEP6_DIR = Path("./Step6_PCA_WeightingFunctions")
STEP6_COMBINED_CSV = STEP6_DIR / "DeltaI_weighting_functions_all.csv"

STEP5_DELTA_C_CSV = Path("./Step5_Perturbation_Joint/Joint_k99/delta_c_used_k99.csv")

OUT_DIR = Path("./Step7_InfoContent_Analysis")  # no timestamp

# Sliding-window defaults (μm)
WINDOW_WIDTH_UM_DEFAULT = 0.5
WINDOW_STRIDE_UM_DEFAULT = 0.1

# Ranking settings
TOP_N_BY_ENERGY = 25  # requirement: 25 top windows by energy
TOP_K_PER_PC = 5      # requirement: top 5 windows per PC (not restricted by dominance)
ENFORCE_MIN_CENTER_SEPARATION_UM = 0.5 * WINDOW_WIDTH_UM_DEFAULT  # for energy-only top windows

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
    Loads the combined ΔI(λ) CSV produced by Step 6.
    Returns:
      - lam [μm]: ascending wavelength grid (index 'lambda_um')
      - deltaI: dict column -> ndarray [Nλ,] (W/(m^2 · sr · μm))
    """
    if not path.exists():
        raise FileNotFoundError(f"Step 6 combined CSV not found: {path}")
    df = pd.read_csv(path, index_col=0)
    lam = df.index.values.astype(float)
    if lam[0] > lam[-1]:
        df = df.iloc[::-1].copy()
        lam = df.index.values.astype(float)
    cols = [c for c in df.columns if c.startswith("PC") and ("plus" in c or "minus" in c)]
    deltaI = {c: df[c].values.astype(float) for c in cols}
    return lam, deltaI


# (S7.2) Function — Title: Build W_i(λ) = 0.5·(ΔI_plus − ΔI_minus)
def build_pc_halfdiff_sensitivity(
    deltaI: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Returns dict of W_i(λ): 'PC{i}_W' -> ndarray [Nλ,]
    """
    pc_W: Dict[str, np.ndarray] = {}
    for i in range(1, 10):
        kp = f"PC{i}_plus"
        km = f"PC{i}_minus"
        if kp not in deltaI or km not in deltaI:
            raise KeyError(f"Missing ΔI columns for PC{i}: {kp} or {km}")
        pc_W[f"PC{i}_W"] = 0.5 * (deltaI[kp] - deltaI[km])
    return pc_W


# (S7.3) Function — Title: Load Δc_i values from Step 5 CSV
def load_delta_c_values(path: Path) -> Dict[int, float]:
    """
    Attempts to parse Δc_i from CSV.
    Accepts formats:
      - Two columns: 'PC','delta_c' (or 'value'), rows: PC1..PC9
      - Single row with columns named 'PC1'..'PC9'
      - Single column with header 'PC1'..'PC9' and one row of values
    Returns dict: {i: Δc_i} for i=1..9
    """
    if not path.exists():
        raise FileNotFoundError(f"Δc CSV not found: {path}")
    df = pd.read_csv(path)

    # Case A: 'PC' + ('delta_c' or 'value')
    colnames = [c.lower() for c in df.columns]
    if "pc" in colnames and ("delta_c" in colnames or "value" in colnames):
        pc_col = df.columns[colnames.index("pc")]
        val_col = df.columns[colnames.index("delta_c")] if "delta_c" in colnames else df.columns[colnames.index("value")]
        d = {}
        for _, row in df.iterrows():
            pc_str = str(row[pc_col]).strip().upper()
            if pc_str.startswith("PC"):
                idx = int(pc_str.replace("PC", ""))
                d[idx] = float(row[val_col])
        if len(d) == 9:
            return d

    # Case B: single row with columns PC1..PC9
    cols_upper = [c.upper() for c in df.columns]
    if all(f"PC{i}" in cols_upper for i in range(1, 10)):
        row = df.iloc[0]
        d = {i: float(row[df.columns[cols_upper.index(f"PC{i}")]]) for i in range(1, 10)}
        return d

    # Case C: single column with headers PC1..PC9 and one row
    if df.shape[0] == 1 and df.shape[1] >= 9:
        try:
            d = {i: float(df.iloc[0][f"PC{i}"]) for i in range(1, 10)}
            return d
        except Exception:
            pass

    raise ValueError(f"Unrecognized Δc CSV format: {path}. Please provide a table I can parse.")


# (S7.4) Function — Title: Build U_i(λ) = W_i(λ) / Δc_i (unit-coefficient sensitivity)
def build_pc_unit_sensitivity(
    pc_W: Dict[str, np.ndarray],
    delta_c: Dict[int, float]
) -> Dict[str, np.ndarray]:
    """
    Returns dict of U_i(λ): 'PC{i}_U' -> ndarray [Nλ,]
    """
    pc_U: Dict[str, np.ndarray] = {}
    for i in range(1, 10):
        keyW = f"PC{i}_W"
        if i not in delta_c:
            raise KeyError(f"Missing Δc for PC{i}")
        denom = float(delta_c[i])
        if denom == 0.0:
            raise ZeroDivisionError(f"Δc_i is zero for PC{i}")
        pc_U[f"PC{i}_U"] = pc_W[keyW] / denom
    return pc_U


# (S7.5) Function — Title: Sensitivity curves S_rms(λ), S_max(λ)
def compute_sensitivity_curves(fields: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a dict of spectral fields f_k(λ) -> arrays [Nλ,], compute:
      S_rms(λ) = sqrt(mean_k f_k(λ)^2), S_max(λ) = max_k |f_k(λ)|
    """
    arr = np.vstack([v for v in fields.values()]).T
    S_rms = np.sqrt(np.mean(arr**2, axis=1))
    S_max = np.max(np.abs(arr), axis=1)
    return S_rms, S_max


# (S7.6) Function — Title: Sliding-window integration and metrics for a set of curves
def sliding_window_metrics(
    lam: np.ndarray,
    fields: Dict[str, np.ndarray],
    window_width_um: float,
    window_stride_um: float,
    min_points: int = 5
) -> pd.DataFrame:
    """
    Integrate each curve F_i(λ) over sliding windows:
      V_i = ∫ F_i(λ) dλ (trapezoid)
    Compute window metrics:
      - E_L2 = ||V||_2
      - dominant_pc, dominant_sign
      - dominance_index = max_i |V_i| / Σ_i |V_i|
      - opposition_fraction = fraction of significant, non-dominant components whose sign opposes the dominant sign
    Store V_PC1..V_PC9 for downstream per-PC scoring.
    Returns DataFrame with window-wise rows and columns including V_i.
    """
    lam = lam.astype(float)
    lam_min, lam_max = lam.min(), lam.max()
    centers = np.arange(lam_min + 0.5 * window_width_um, lam_max - 0.5 * window_width_um + 1e-12, window_stride_um)

    # Ensure keys are PC1..PC9 in numeric order for stacking
    keys = sorted(fields.keys(), key=lambda k: int(''.join(filter(str.isdigit, k))))
    F = np.vstack([fields[k] for k in keys]).T  # [Nλ, 9]
    if F.shape[1] != 9:
        raise ValueError("Expected 9 components (PC1..PC9).")

    rows = []
    for c in centers:
        a, b = c - 0.5 * window_width_um, c + 0.5 * window_width_um
        idx = np.where((lam >= a) & (lam <= b))[0]
        if idx.size < min_points:
            continue
        lam_w = lam[idx]
        F_w = F[idx, :]  # [n, 9]
        V = np.trapz(F_w, lam_w, axis=0)  # [9,]
        absV = np.abs(V)
        sum_absV = np.sum(absV)
        E = float(np.linalg.norm(V, ord=2))

        if sum_absV == 0.0:
            dom_idx = -1
            dom_sign = 0
            DI = 0.0
            opp_frac = 0.0
        else:
            dom_idx = int(np.argmax(absV))  # 0-based
            dom_sign = int(np.sign(V[dom_idx]))
            DI = float(absV[dom_idx] / sum_absV)
            thr = 0.01 * absV[dom_idx]
            significant = absV >= thr
            signs = np.sign(V)
            mask_not_dom = np.ones_like(significant, dtype=bool)
            mask_not_dom[dom_idx] = False
            denom = int(np.sum(significant & mask_not_dom))
            if denom <= 0:
                opp_frac = 0.0
            else:
                opp_frac = float(np.sum((signs == -dom_sign) & significant & mask_not_dom) / denom)

        row = {
            "lambda_center_um": float(c),
            "lambda_start_um": float(a),
            "lambda_end_um": float(b),
            "num_points": int(idx.size),
            "E_L2": E,
            "dominant_pc": f"PC{dom_idx+1}" if dom_idx >= 0 else "None",
            "dominant_sign": int(dom_sign),
            "dominance_index": DI,
            "opposition_fraction": opp_frac,
        }
        for i in range(9):
            row[f"V_PC{i+1}"] = float(V[i])
        row["sum_absV"] = float(sum_absV)  # useful for per-PC DI_i
        rows.append(row)

    df = pd.DataFrame(rows)
    df.index.name = "window_id"
    return df


# (S7.7) Function — Title: Non-overlapping Top-N selection by score
def select_top_windows_nonoverlap(
    df_windows: pd.DataFrame,
    score_col: str,
    top_n: int,
    min_separation_um: float
) -> pd.DataFrame:
    """
    Greedy selection by descending score with minimum center separation.
    """
    df = df_windows.sort_values(score_col, ascending=False).copy()
    selected = []
    centers = []
    for idx, row in df.iterrows():
        c = row["lambda_center_um"]
        if all(abs(c - sc) >= min_separation_um for sc in centers):
            selected.append(idx); centers.append(c)
        if len(selected) >= top_n:
            break
    return df.loc[selected]


# (S7.8) Function — Title: Top-K windows per PC by continuous per-PC score (no dominance requirement)
def select_top_k_per_pc(
    df_windows: pd.DataFrame,
    top_k: int
) -> pd.DataFrame:
    """
    For each PC i:
      - Compute DI_i(window) = |V_i| / Σ_j |V_j|
      - score_i(window) = E_L2(window) * DI_i(window)
      - Select top_k windows by score_i (no dominance restriction).
    Returns concatenated DataFrame with additional columns:
      ['target_pc','DI_target','V_target','score_per_pc']
    """
    if df_windows.empty:
        return pd.DataFrame()

    rows = []
    for i in range(1, 10):
        pc_col = f"V_PC{i}"
        if pc_col not in df_windows.columns:
            raise KeyError(f"Missing column {pc_col} in window metrics.")
        dfi = df_windows.copy()
        denom = dfi["sum_absV"].replace(0.0, np.nan)
        DI_i = np.abs(dfi[pc_col]) / denom
        DI_i = DI_i.fillna(0.0)
        score = dfi["E_L2"] * DI_i
        dfi = dfi.assign(target_pc=f"PC{i}", DI_target=DI_i.values, V_target=dfi[pc_col].values, score_per_pc=score.values)
        dfi = dfi.sort_values("score_per_pc", ascending=False).head(top_k)
        rows.append(dfi)

    out = pd.concat(rows, axis=0)
    # Keep original index (window_id); add PC-level multi-key via a column
    return out


# (S7.9) Function — Title: Pairwise correlation among a set of curves
def pairwise_correlation(fields: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Pearson correlation among provided curves across λ.
    """
    keys = sorted(fields.keys(), key=lambda k: int(''.join(filter(str.isdigit, k))))
    mat = np.vstack([fields[k] for k in keys])
    C = np.corrcoef(mat)
    return pd.DataFrame(C, index=keys, columns=keys)


# (S7.10) Function — Title: Save CSVs (headers + index)
def save_csvs_step7(
    out_dir: Path,
    lam: np.ndarray,
    pc_W: Dict[str, np.ndarray],
    pc_U: Dict[str, np.ndarray],
    S_W: Tuple[np.ndarray, np.ndarray],
    S_U: Tuple[np.ndarray, np.ndarray],
    df_win_W: pd.DataFrame,
    df_win_U: pd.DataFrame,
    df_top_U: pd.DataFrame,
    df_top_perpc_U: pd.DataFrame,
    corr_W: pd.DataFrame,
    corr_U: pd.DataFrame
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    lam_index = pd.Index(lam, name="lambda_um")

    # Spectral fields
    df_W = pd.DataFrame({k: v for k, v in pc_W.items()}, index=lam_index)
    f_W = out_dir / "PC_W_halfdiff_sensitivity.csv"
    df_W.to_csv(f_W); files["PC_W_halfdiff_sensitivity.csv"] = str(f_W)

    df_U = pd.DataFrame({k: v for k, v in pc_U.items()}, index=lam_index)
    f_U = out_dir / "PC_U_unit_sensitivity.csv"
    df_U.to_csv(f_U); files["PC_U_unit_sensitivity.csv"] = str(f_U)

    # Sensitivity curves
    df_SW = pd.DataFrame({"S_rms_W": S_W[0], "S_max_W": S_W[1]}, index=lam_index)
    f_SW = out_dir / "S_curves_W.csv"
    df_SW.to_csv(f_SW); files["S_curves_W.csv"] = str(f_SW)

    df_SU = pd.DataFrame({"S_rms_U": S_U[0], "S_max_U": S_U[1]}, index=lam_index)
    f_SU = out_dir / "S_curves_U.csv"
    df_SU.to_csv(f_SU); files["S_curves_U.csv"] = str(f_SU)

    # Windows
    f_winW = out_dir / "window_metrics_W.csv"
    df_win_W.to_csv(f_winW); files["window_metrics_W.csv"] = str(f_winW)

    f_winU = out_dir / "window_metrics_U.csv"
    df_win_U.to_csv(f_winU); files["window_metrics_U.csv"] = str(f_winU)

    f_topU = out_dir / "top_windows_by_energy_U.csv"
    df_top_U.to_csv(f_topU); files["top_windows_by_energy_U.csv"] = str(f_topU)

    if df_top_perpc_U is not None and not df_top_perpc_U.empty:
        f_topPCU = out_dir / "top_windows_per_pc_U.csv"
        df_top_perpc_U.to_csv(f_topPCU); files["top_windows_per_pc_U.csv"] = str(f_topPCU)

    # Correlations
    f_corrW = out_dir / "PC_pairwise_correlation_W.csv"
    corr_W.to_csv(f_corrW); files["PC_pairwise_correlation_W.csv"] = str(f_corrW)

    f_corrU = out_dir / "PC_pairwise_correlation_U.csv"
    corr_U.to_csv(f_corrU); files["PC_pairwise_correlation_U.csv"] = str(f_corrU)

    return files


# (S7.11) Function — Title: Plots for W and U analyses (include energy and per-PC windows)
def make_plots_step7(
    out_dir: Path,
    lam: np.ndarray,
    pc_W: Dict[str, np.ndarray],
    pc_U: Dict[str, np.ndarray],
    S_W: Tuple[np.ndarray, np.ndarray],
    S_U: Tuple[np.ndarray, np.ndarray],
    df_top_U: pd.DataFrame,
    df_top_perpc_U: pd.DataFrame
) -> List[str]:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[str] = []

    # Overlay W
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(1, 10):
        ax.plot(lam, pc_W[f"PC{i}_W"], lw=1.2, label=f"PC{i}_W")
    ax.axhline(0, color="k", lw=1.0, alpha=0.6)
    ax.set_xlabel("Wavelength λ (μm)"); ax.set_ylabel("W_i(λ) [W/(m²·sr·μm)]")
    ax.set_title("Per-PC Half-difference Sensitivity W_i(λ)")
    ax.legend(ncol=3, fontsize=8)
    p1 = plots_dir / "overlay_PC_W.png"; fig.tight_layout(); fig.savefig(p1); plt.close(fig); out_paths.append(str(p1))

    # Overlay U
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(1, 10):
        ax.plot(lam, pc_U[f"PC{i}_U"], lw=1.4, label=f"PC{i}_U")
    ax.axhline(0, color="k", lw=1.0, alpha=0.6)
    ax.set_xlabel("Wavelength λ (μm)"); ax.set_ylabel("U_i(λ) [W/(m²·sr·μm)]")
    ax.set_title("Per-PC Unit-coefficient Sensitivity U_i(λ) = W_i(λ)/Δc_i")
    ax.legend(ncol=3, fontsize=8)
    p2 = plots_dir / "overlay_PC_U.png"; fig.tight_layout(); fig.savefig(p2); plt.close(fig); out_paths.append(str(p2))

    # Sensitivity curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lam, S_W[0], color="darkred", lw=1.6, label="S_rms (W)")
    ax.plot(lam, S_W[1], color="firebrick", lw=1.0, alpha=0.8, label="S_max (W)")
    ax.plot(lam, S_U[0], color="navy", lw=1.6, label="S_rms (U)")
    ax.plot(lam, S_U[1], color="royalblue", lw=1.0, alpha=0.8, label="S_max (U)")
    ax.set_xlabel("Wavelength λ (μm)"); ax.set_ylabel("Sensitivity [W/(m²·sr·μm)]")
    ax.set_title("Sensitivity Curves — W vs U")
    ax.legend()
    p3 = plots_dir / "sensitivity_curves_W_vs_U.png"; fig.tight_layout(); fig.savefig(p3); plt.close(fig); out_paths.append(str(p3))

    # U overlay with Top-25 energy windows shaded (gold)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(1, 10):
        ax.plot(lam, pc_U[f"PC{i}_U"], lw=1.0, label=f"PC{i}_U")
    if df_top_U is not None and not df_top_U.empty:
        ymin = min(np.min(v) for v in pc_U.values())
        ymax = max(np.max(v) for v in pc_U.values())
        for _, row in df_top_U.iterrows():
            ax.axvspan(row["lambda_start_um"], row["lambda_end_um"], color="gold", alpha=0.28, lw=0)
        ax.set_ylim(ymin*1.05 if ymin<0 else ymin*0.95, ymax*1.05)
    ax.axhline(0, color="k", lw=1.0, alpha=0.6)
    ax.set_xlabel("Wavelength λ (μm)"); ax.set_ylabel("U_i(λ) [W/(m²·sr·μm)]")
    ax.set_title("Top-25 Energy Windows (U-based) shaded over U_i(λ)")
    ax.legend(ncol=3, fontsize=8)
    p4 = plots_dir / "overlay_U_with_top_windows_energy.png"; fig.tight_layout(); fig.savefig(p4); plt.close(fig); out_paths.append(str(p4))

    # U overlay with BOTH Top-25 energy (gold) and Top-5 per PC windows (colored)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(1, 10):
        ax.plot(lam, pc_U[f"PC{i}_U"], lw=1.0, label=f"PC{i}_U")
    ymin = min(np.min(v) for v in pc_U.values())
    ymax = max(np.max(v) for v in pc_U.values())
    # Shade energy windows
    if df_top_U is not None and not df_top_U.empty:
        for _, row in df_top_U.iterrows():
            ax.axvspan(row["lambda_start_um"], row["lambda_end_um"], color="gold", alpha=0.24, lw=0)
    # Shade per-PC windows (distinct colors)
    cmap = plt.get_cmap("tab10")
    legend_handles = [mpatches.Patch(color="gold", alpha=0.24, label="Top-25 by Energy")]
    if df_top_perpc_U is not None and not df_top_perpc_U.empty:
        for pc_i in range(1, 10):
            dfi = df_top_perpc_U[df_top_perpc_U["target_pc"] == f"PC{pc_i}"]
            color = cmap((pc_i - 1) % 10)
            for _, row in dfi.iterrows():
                ax.axvspan(row["lambda_start_um"], row["lambda_end_um"], color=color, alpha=0.18, lw=0)
            legend_handles.append(mpatches.Patch(color=color, alpha=0.18, label=f"Top-5 for PC{pc_i}"))
    ax.set_ylim(ymin*1.05 if ymin<0 else ymin*0.95, ymax*1.05)
    ax.axhline(0, color="k", lw=1.0, alpha=0.6)
    ax.set_xlabel("Wavelength λ (μm)"); ax.set_ylabel("U_i(λ) [W/(m²·sr·μm)]")
    ax.set_title("Top Windows: Energy (gold) + Top-5 Per-PC (colored) over U_i(λ)")
    ax.legend(handles=legend_handles, ncol=2, fontsize=8, frameon=True)
    p5 = plots_dir / "overlay_U_with_top_windows_energy_and_perPC.png"; fig.tight_layout(); fig.savefig(p5); plt.close(fig); out_paths.append(str(p5))

    return out_paths


# (S7.12) Function — Title: Write inventory log for Step 7 (normalized with new selections)
def write_inventory_log_step7(
    out_dir: Path,
    step6_csv: Path,
    step5_deltac_csv: Path,
    window_width_um: float,
    window_stride_um: float,
    files_map: Dict[str, str],
    plot_paths: List[str]
) -> str:
    log_path = out_dir / "Step7_InfoContent_inventory.log"
    lines: List[str] = []
    lines.append("Ground-based Atmospheric Profiles — Step 7: Information Content Analysis (Δc-normalized)")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"  - Step 6 ΔI combined CSV: {str(step6_csv.resolve())}")
    lines.append(f"  - Step 5 Δc CSV:          {str(step5_deltac_csv.resolve())}")
    lines.append("")
    lines.append("Core relations:")
    lines.append("  - W_i(λ) = 0.5 · (ΔI_i,plus(λ) − ΔI_i,minus(λ)) ≈ Δc_i · ∂I(λ)/∂c_i")
    lines.append("  - U_i(λ) = W_i(λ) / Δc_i ≈ ∂I(λ)/∂c_i  (unit-coefficient sensitivity)")
    lines.append("  - Sliding window [λ_c − Δλ/2, λ_c + Δλ/2]: V_i = ∫ F_i(λ) dλ, F = W or U")
    lines.append("    Metrics: E_L2 = ||V||_2; dominance_index = max_i |V_i| / Σ_i |V_i|;")
    lines.append("             opposition_fraction = fraction of significant (≥1% of dominant) non-dominant components")
    lines.append("             whose sign opposes the dominant sign.")
    lines.append(f"  - Window settings: width Δλ = {window_width_um:.3f} μm, stride = {window_stride_um:.3f} μm.")
    lines.append("")
    lines.append("Selection rules implemented:")
    lines.append("  - Top-25 by energy (U-based): non-overlapping windows by descending E_L2.")
    lines.append("  - Top-5 per PC (U-based): for each PC i, score_i = E_L2 × DI_i with DI_i = |V_i|/Σ|V_j|;")
    lines.append("    select 5 highest-scoring windows (no dominance requirement).")
    lines.append("")
    lines.append("Outputs (all CSVs include headers; λ-indexed tables use 'lambda_um'):")
    for k, v in files_map.items():
        if k == "PC_U_unit_sensitivity.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: λ-indexed per-PC unit-coefficient sensitivities U_i(λ) = W_i(λ)/Δc_i.")
        elif k == "S_curves_U.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: λ-indexed S_rms and S_max across U_i(λ).")
        elif k == "window_metrics_U.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: Sliding-window metrics on U_i, includes V_PC1..V_PC9 and sum_absV for per-PC scoring.")
        elif k == "top_windows_by_energy_U.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: Top-25 non-overlapping windows by E_L2 (U-based); candidate high-signal bands.")
        elif k == "top_windows_per_pc_U.csv":
            lines.append(f"  - {k}: {v}")
            lines.append("    Meaning: For each PC, Top-5 windows by score_i = E_L2 × (|V_i|/Σ|V_j|); no dominance constraint.")
        else:
            lines.append(f"  - {k}: {v}")
    lines.append("")
    if plot_paths:
        lines.append("Figures:")
        for p in plot_paths:
            lines.append(f"  - {p}")
        lines.append("  Meanings:")
        lines.append("    - overlay_PC_W.png: Per-PC W_i(λ) = 0.5·(ΔI_plus − ΔI_minus).")
        lines.append("    - overlay_PC_U.png: Per-PC U_i(λ) = W_i(λ)/Δc_i (unit-coefficient).")
        lines.append("    - sensitivity_curves_W_vs_U.png: S_rms, S_max vs λ for W and U.")
        lines.append("    - overlay_U_with_top_windows_energy.png: U_i(λ) with Top-25 E_L2 windows shaded (gold).")
        lines.append("    - overlay_U_with_top_windows_energy_and_perPC.png: U_i(λ) with Top-25 energy (gold) and Top-5 per PC (colored) shaded.")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - Δc normalization removes artificial dominance by PC1 due to larger bin width.")
    lines.append("  - No detector system effects are included in Step 7.")
    lines.append("  - Outputs are written without timestamps for reproducibility.")
    lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return str(log_path)


# Main runner — Title: run_step7_information_content_main (no CLI parsing)
def run_step7_information_content_main():
    """
    Steps:
      1) Load Step 6 ΔI(λ).
      2) Build W_i(λ) half-difference sensitivities.
      3) Load Δc_i and build U_i(λ) = W_i/Δc_i (unit-coefficient).
      4) Compute sensitivity curves for W and U.
      5) Sliding-window metrics for W and U.
      6) Top-25 non-overlapping windows by energy using U-based metrics.
      7) Top-5 per PC by score_i = E_L2 × (|V_i| / Σ|V_j|); no dominance requirement.
      8) Pairwise correlations (W and U).
      9) Save CSVs and plots; write an inventory log.
    """
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) ΔI(λ)
    lam, deltaI = load_step6_deltaI_combined(STEP6_COMBINED_CSV)

    # 2) W_i(λ)
    pc_W = build_pc_halfdiff_sensitivity(deltaI)

    # 3) Δc_i and U_i(λ)
    delta_c = load_delta_c_values(STEP5_DELTA_C_CSV)
    pc_U = build_pc_unit_sensitivity(pc_W, delta_c)

    # 4) Sensitivity curves
    S_W = compute_sensitivity_curves(pc_W)
    S_U = compute_sensitivity_curves(pc_U)

    # 5) Sliding-window metrics
    df_win_W = sliding_window_metrics(
        lam=lam, fields=pc_W,
        window_width_um=WINDOW_WIDTH_UM_DEFAULT,
        window_stride_um=WINDOW_STRIDE_UM_DEFAULT,
        min_points=5
    )
    df_win_U = sliding_window_metrics(
        lam=lam, fields=pc_U,
        window_width_um=WINDOW_WIDTH_UM_DEFAULT,
        window_stride_um=WINDOW_STRIDE_UM_DEFAULT,
        min_points=5
    )

    # 6) Top-25 windows by energy (U-based, non-overlapping)
    df_top_U = select_top_windows_nonoverlap(
        df_windows=df_win_U,
        score_col="E_L2",
        top_n=TOP_N_BY_ENERGY,
        min_separation_um=ENFORCE_MIN_CENTER_SEPARATION_UM
    )

    # 7) Top-5 per PC (U-based, no dominance requirement)
    df_top_perpc_U = select_top_k_per_pc(
        df_windows=df_win_U,
        top_k=TOP_K_PER_PC
    )

    # 8) Pairwise correlations
    corr_W = pairwise_correlation(pc_W)
    corr_U = pairwise_correlation(pc_U)

    # 9) Plots (include both energy-only and combined overlays)
    plot_paths = make_plots_step7(
        out_dir=out_dir,
        lam=lam,
        pc_W=pc_W,
        pc_U=pc_U,
        S_W=S_W,
        S_U=S_U,
        df_top_U=df_top_U,
        df_top_perpc_U=df_top_perpc_U
    )

    # 10) Save CSVs
    files_map = save_csvs_step7(
        out_dir=out_dir,
        lam=lam,
        pc_W=pc_W,
        pc_U=pc_U,
        S_W=S_W,
        S_U=S_U,
        df_win_W=df_win_W,
        df_win_U=df_win_U,
        df_top_U=df_top_U,
        df_top_perpc_U=df_top_perpc_U,
        corr_W=corr_W,
        corr_U=corr_U
    )

    # 11) Log
    log_path = write_inventory_log_step7(
        out_dir=out_dir,
        step6_csv=STEP6_COMBINED_CSV,
        step5_deltac_csv=STEP5_DELTA_C_CSV,
        window_width_um=WINDOW_WIDTH_UM_DEFAULT,
        window_stride_um=WINDOW_STRIDE_UM_DEFAULT,
        files_map=files_map,
        plot_paths=plot_paths
    )

    print("Step 7 (Δc-normalized) completed.")
    print(f"- Outputs dir: {str(out_dir.resolve())}")
    print(f"- Inventory log: {log_path}")


# Entry point for VSCode "Run Python File"
if __name__ == "__main__":
    run_step7_information_content_main()