# (J2) Step 2 — Joint PCA of Temperature and Dew-point (VSCode-ready)
# Title: step2_joint_pca.py
# Author: [Your Name]
# Date: 2025-08-28
# Description:
#   Jointly analyzes temperature (T) and dew-point (Td) vertical profiles via PCA.
#   - Builds a joint centered feature matrix X_joint (888 × 42) by concatenating T (21) and Td (21).
#   - Offers scaling options to control each block’s contribution:
#       * 'none'   : center-only (uses your Step 1 centered inputs as-is).
#       * 'zscore' : per-feature standardization (dimensionless loadings).
#       * 'block'  : equalize total variance of T and Td blocks.
#   - Computes PCA, selects k for cumulative explained variance targets (e.g., 95%, 99%).
#   - Saves W_joint (42 × p), C (888 × p), and split loadings W_T (21 × p), W_Td (21 × p).
#   - Plots:
#       * Scree and cumulative explained variance (joint).
#       * PC loadings for T and Td up to k at 99% cumulative explained variance ("99% variance mode").
#
# Core joint PCA relation (readable fonts and meanings):
#   - C = X_joint_centered · W_joint
#     where
#       C                 : (n_samples × p) scores (principal coefficients).
#       X_joint_centered  : (n_samples × 42) joint centered (and optionally scaled) matrix.
#                           The first 21 columns are T (0–2 km), next 21 columns are Td (0–2 km).
#       W_joint           : (42 × p) principal loadings (PCs as columns).
#     Split loadings:
#       W_T  = W_joint[0:21, :]   (temperature part of each PC)
#       W_Td = W_joint[21:42, :]  (dew-point part of each PC)
#
# How to run (main program):
#   - Edit the configuration in Step2JointConfig() as needed.
#   - From VSCode: Run this file. From Python: call run_step2_joint().
#
# Local functions (titles and numbering):
#   (J2.1) read_vector_csv_robust    : Robust 1D vector CSV reader (handles row/column vectors with/without headers).
#   (J2.2) load_step1_outputs        : Load Step 1 centered matrices and height vector.
#   (J2.3) build_joint_matrix        : Concatenate T and Td and apply chosen scaling mode.
#   (J2.4) run_pca                   : Execute PCA and return loadings, scores, explained variance metrics.
#   (J2.5) decide_k                  : Choose minimal k for each cumulative explained variance target.
#   (J2.6) save_joint_outputs        : Save W_joint, W_T, W_Td, C, and variance tables to CSV.
#   (J2.7) plot_joint_diagnostics    : Save scree and cumulative plots.
#   (J2.8) plot_joint_pc_loadings    : Plot PC loadings (T and Td vs height) up to k99 (“99% variance mode”).
#   (J2.9) write_log_joint           : Write an inventory log describing shapes, scaling, k’s, and output files.
#   run_step2_joint                  : Main runner that ties everything together.
#
# Notes:
#   - Uses Step 1 outputs (centered T and Td): no detector system or radiance inputs at this stage.
#   - H (length 61) is used to label the first 21 heights (0–2 km) on plots and CSV indices.

from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math


# =========================
# Configuration (edit here)
# =========================

@dataclass
class Step2JointConfig:
    # Path to Step 1 processed outputs
    step1_dir: Path = Path("./Step1_Processed")
    # Base output directory for Step 2 Joint PCA; a timestamped subfolder will be created inside
    out_dir_base: Path = Path(r"C:\Users\24573\OneDrive - The University of Hong Kong\PhD\Project\atmospheric sounding\prior-analysis")
    # Cumulative explained variance targets
    variance_targets: List[float] = None
    # Scaling mode for joint analysis: 'none' | 'zscore' | 'block'
    scaling_mode: str = 'zscore'
    # Whether to save plots
    save_plots: bool = True
    # Subfolder name (auto-timestamped if None)
    out_subfolder_name: str = None

    def __post_init__(self):
        if self.variance_targets is None:
            self.variance_targets = [0.95, 0.99]
        if self.out_subfolder_name is None:
            ts = datetime.datetime.now().strftime("Step2_JointPCA_%Y%m%d_%H%M%S")
            self.out_subfolder_name = ts


# (J2.1) Function: read_vector_csv_robust
def read_vector_csv_robust(path: Path, expected_len: int) -> np.ndarray:
    """
    Robustly read a 1D vector from CSV (row/column, with/without headers).
    Returns np.ndarray of shape (expected_len,).
    """
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    import pandas as pd
    def numeric_flat(df: pd.DataFrame) -> np.ndarray:
        df_num = df.apply(pd.to_numeric, errors='coerce')
        arr = df_num.values.ravel()
        return arr[~np.isnan(arr)]

    try:
        df1 = pd.read_csv(path, index_col=0)
        if df1.shape[1] == 0:
            df1 = pd.read_csv(path)
    except Exception:
        df1 = pd.read_csv(path)

    if df1.shape[0] == expected_len and df1.shape[1] >= 1:
        col0 = pd.to_numeric(df1.iloc[:, 0], errors='coerce').values
        if np.count_nonzero(~np.isnan(col0)) == expected_len:
            return col0.astype(float)

    if df1.shape[1] == expected_len and df1.shape[0] >= 1:
        row0 = pd.to_numeric(df1.iloc[0, :], errors='coerce').values
        if np.count_nonzero(~np.isnan(row0)) == expected_len:
            return row0.astype(float)

    flat1 = numeric_flat(df1)
    if flat1.size >= expected_len:
        return flat1[:expected_len].astype(float)

    df2 = pd.read_csv(path, header=None)
    if df2.shape[0] == expected_len and df2.shape[1] >= 1:
        col0 = pd.to_numeric(df2.iloc[:, 0], errors='coerce').values
        if np.count_nonzero(~np.isnan(col0)) == expected_len:
            return col0.astype(float)

    if df2.shape[1] == expected_len and df2.shape[0] >= 1:
        row0 = pd.to_numeric(df2.iloc[0, :], errors='coerce').values
        if np.count_nonzero(~np.isnan(row0)) == expected_len:
            return row0.astype(float)

    flat2 = numeric_flat(df2)
    if flat2.size >= expected_len:
        return flat2[:expected_len].astype(float)

    raise ValueError(f"Could not extract a length-{expected_len} numeric vector from: {path}")


# (J2.2) Function: load_step1_outputs
def load_step1_outputs(step1_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load Step 1 centered matrices and vectors.
    Returns a dict with:
      - 'X_T_centered'  : DataFrame (888 × 21)
      - 'X_Td_centered' : DataFrame (888 × 21)
      - 'mu_T'          : np.ndarray (21,)
      - 'mu_Td'         : np.ndarray (21,)
      - 'H'             : np.ndarray (61,)
    """
    files = {
        'X_T_centered': step1_dir / "X_T_centered.csv",
        'X_Td_centered': step1_dir / "X_Td_centered.csv",
        'mu_T': step1_dir / "mu_T.csv",
        'mu_Td': step1_dir / "mu_Td.csv",
        'H': step1_dir / "H.csv",
    }
    for k, f in files.items():
        if not f.exists():
            raise FileNotFoundError(f"Required Step 1 file not found: {f}")

    X_T_centered = pd.read_csv(files['X_T_centered'], index_col=0).apply(pd.to_numeric, errors='coerce')
    X_Td_centered = pd.read_csv(files['X_Td_centered'], index_col=0).apply(pd.to_numeric, errors='coerce')
    mu_T = read_vector_csv_robust(files['mu_T'], expected_len=21)
    mu_Td = read_vector_csv_robust(files['mu_Td'], expected_len=21)
    H = read_vector_csv_robust(files['H'], expected_len=61)

    return {
        'X_T_centered': X_T_centered,
        'X_Td_centered': X_Td_centered,
        'mu_T': mu_T,
        'mu_Td': mu_Td,
        'H': H,
    }


# (J2.3) Function: build_joint_matrix
def build_joint_matrix(
    X_Tc: pd.DataFrame,
    X_Tdc: pd.DataFrame,
    scaling_mode: str = 'zscore'
) -> Dict[str, object]:
    """
    Build joint matrix from centered T and Td:
      - X_Tc: (n × 21), centered temperature profiles
      - X_Tdc: (n × 21), centered dew-point profiles
      - scaling_mode: 'none' | 'zscore' | 'block'

    Returns dict with:
      - 'X_joint'      : (n × 42) joint centered and scaled matrix
      - 'scaling_info' : dict describing applied scaling
      - 'feature_cols' : list of 42 column labels (e.g., 'T_1', ..., 'T_21', 'Td_1', ..., 'Td_21')
    """
    if X_Tc.shape != X_Tdc.shape:
        raise ValueError(f"T and Td shapes must match. Got {X_Tc.shape} vs {X_Tdc.shape}")
    n, m = X_Tc.shape  # n=888, m=21

    # Ensure numerical types and zero mean (they should already be centered from Step 1)
    X_Tc_vals = X_Tc.values.astype(float)
    X_Tdc_vals = X_Tdc.values.astype(float)

    scaling_info = {'mode': scaling_mode}
    # Per-feature standard deviations for z-score (using centered data; mean=0)
    if scaling_mode.lower() == 'zscore':
        s_T = X_Tc_vals.std(axis=0, ddof=1)
        s_Td = X_Tdc_vals.std(axis=0, ddof=1)
        # Avoid division by zero
        s_T[s_T == 0] = 1.0
        s_Td[s_Td == 0] = 1.0
        X_Tc_scaled = X_Tc_vals / s_T
        X_Tdc_scaled = X_Tdc_vals / s_Td
        scaling_info.update({
            's_T_per_feature': s_T,
            's_Td_per_feature': s_Td
        })
    elif scaling_mode.lower() == 'block':
        # Equalize total variance of T and Td blocks
        var_T_feats = X_Tc_vals.var(axis=0, ddof=1)  # length 21
        var_Td_feats = X_Tdc_vals.var(axis=0, ddof=1)
        sum_var_T = float(np.sum(var_T_feats))
        sum_var_Td = float(np.sum(var_Td_feats))
        # Avoid divide-by-zero
        sum_var_T = sum_var_T if sum_var_T > 0 else 1.0
        sum_var_Td = sum_var_Td if sum_var_Td > 0 else 1.0
        alpha_T = 1.0 / math.sqrt(sum_var_T)
        alpha_Td = 1.0 / math.sqrt(sum_var_Td)
        X_Tc_scaled = alpha_T * X_Tc_vals
        X_Tdc_scaled = alpha_Td * X_Tdc_vals
        scaling_info.update({
            'alpha_T': alpha_T,
            'alpha_Td': alpha_Td,
            'sum_var_T': sum_var_T,
            'sum_var_Td': sum_var_Td
        })
    elif scaling_mode.lower() == 'none':
        # Keep physical units and relative variances
        X_Tc_scaled = X_Tc_vals
        X_Tdc_scaled = X_Tdc_vals
    else:
        raise ValueError("scaling_mode must be one of: 'none', 'zscore', 'block'")

    X_joint = np.hstack([X_Tc_scaled, X_Tdc_scaled])  # (n × 42)
    feature_cols = [f"T_{i+1}" for i in range(m)] + [f"Td_{i+1}" for i in range(m)]
    return {
        'X_joint': X_joint,
        'scaling_info': scaling_info,
        'feature_cols': feature_cols
    }


# (J2.4) Function: run_pca
def run_pca(X_centered: np.ndarray, n_components: int = None) -> Dict[str, np.ndarray]:
    """
    PCA on centered (and possibly scaled) X_centered (n_samples × n_features).
    Returns dict with W (n_features × p), C (n_samples × p), and variance metrics.
    """
    n_samples, n_features = X_centered.shape
    if n_components is None:
        n_components = n_features

    pca = PCA(n_components=n_components, svd_solver='full')
    C = pca.fit_transform(X_centered)  # (n_samples × p)
    components_raw = pca.components_   # (p × n_features)
    W = components_raw.T               # (n_features × p)
    ev = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    return {
        'W': W,
        'C': C,
        'explained_variance': ev,
        'explained_variance_ratio': evr,
        'cum_explained_variance_ratio': cum_evr,
        'components_raw': components_raw,
    }


# (J2.5) Function: decide_k
def decide_k(cum_evr: np.ndarray, targets: List[float]) -> Dict[float, int]:
    """
    For each target in targets (e.g., 0.95, 0.99), find minimal k such that cum_evr[k-1] >= target.
    """
    ks = {}
    for t in targets:
        idx = np.searchsorted(cum_evr, t, side='left')
        k = int(idx + 1) if idx < len(cum_evr) else len(cum_evr)
        ks[float(t)] = k
    return ks


# (J2.6) Function: save_joint_outputs
def save_joint_outputs(
    out_dir: Path,
    W: np.ndarray,
    C: np.ndarray,
    ev: np.ndarray,
    evr: np.ndarray,
    cum_evr: np.ndarray,
    ks: Dict[float, int],
    sample_index: pd.Index,
    H_first21: np.ndarray
) -> Dict[str, str]:
    """
    Save joint PCA outputs:
      - W_joint (42 × p), with MultiIndex rows ('var', 'z'), split into W_T and W_Td (each 21 × p)
      - C (888 × p), scores
      - explained variance tables
      - reduced versions for k at each target
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = W.shape[1]
    pc_cols = [f"PC{i+1}" for i in range(p)]

    # Split W into T and Td blocks
    W_T = W[:21, :]
    W_Td = W[21:42, :]

    # Index for heights (0–2 km)
    df_W_T = pd.DataFrame(W_T, index=H_first21, columns=pc_cols)
    df_W_T.index.name = "H_first21"
    df_W_Td = pd.DataFrame(W_Td, index=H_first21, columns=pc_cols)
    df_W_Td.index.name = "H_first21"

    # Scores with sample indices
    df_C = pd.DataFrame(C, index=sample_index, columns=pc_cols)

    df_ev = pd.DataFrame({"explained_variance": ev}, index=pc_cols)
    df_evr = pd.DataFrame({"explained_variance_ratio": evr}, index=pc_cols)
    df_cumevr = pd.DataFrame({"cumulative_explained_variance_ratio": cum_evr}, index=pc_cols)

    paths = {}

    # Save full outputs
    paths["PCA_joint_W_T_csv"]  = str((out_dir / "PCA_joint_W_T.csv").resolve())
    paths["PCA_joint_W_Td_csv"] = str((out_dir / "PCA_joint_W_Td.csv").resolve())
    paths["PCA_joint_scores_csv"] = str((out_dir / "PCA_joint_scores.csv").resolve())
    paths["PCA_joint_explained_variance_csv"] = str((out_dir / "PCA_joint_explained_variance.csv").resolve())
    paths["PCA_joint_explained_variance_ratio_csv"] = str((out_dir / "PCA_joint_explained_variance_ratio.csv").resolve())
    paths["PCA_joint_cumulative_explained_variance_ratio_csv"] = str((out_dir / "PCA_joint_cumulative_explained_variance_ratio.csv").resolve())

    df_W_T.to_csv(paths["PCA_joint_W_T_csv"])
    df_W_Td.to_csv(paths["PCA_joint_W_Td_csv"])
    df_C.to_csv(paths["PCA_joint_scores_csv"])
    df_ev.to_csv(paths["PCA_joint_explained_variance_csv"])
    df_evr.to_csv(paths["PCA_joint_explained_variance_ratio_csv"])
    df_cumevr.to_csv(paths["PCA_joint_cumulative_explained_variance_ratio_csv"])

    # Save reduced versions for selected k per target
    for t, k in ks.items():
        df_W_T_k = df_W_T.iloc[:, :k]
        df_W_Td_k = df_W_Td.iloc[:, :k]
        df_C_k = df_C.iloc[:, :k]
        paths[f"PCA_joint_W_T_k{int(t*100)}_csv"] = str((out_dir / f"PCA_joint_W_T_k{int(t*100)}.csv").resolve())
        paths[f"PCA_joint_W_Td_k{int(t*100)}_csv"] = str((out_dir / f"PCA_joint_W_Td_k{int(t*100)}.csv").resolve())
        paths[f"PCA_joint_scores_k{int(t*100)}_csv"] = str((out_dir / f"PCA_joint_scores_k{int(t*100)}.csv").resolve())
        df_W_T_k.to_csv(paths[f"PCA_joint_W_T_k{int(t*100)}_csv"])
        df_W_Td_k.to_csv(paths[f"PCA_joint_W_Td_k{int(t*100)}_csv"])
        df_C_k.to_csv(paths[f"PCA_joint_scores_k{int(t*100)}_csv"])

    return paths


# (J2.7) Function: plot_joint_diagnostics
def plot_joint_diagnostics(out_dir: Path, evr: np.ndarray, cum_evr: np.ndarray) -> Dict[str, str]:
    """
    Save joint scree and cumulative explained variance plots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pcs = np.arange(1, len(evr) + 1)

    # Scree plot
    plt.figure(figsize=(7, 5))
    plt.plot(pcs, evr, 'o-', lw=2)
    plt.xlabel("Principal Component (index)")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Joint PCA — Scree Plot (T + Td)")
    plt.grid(True, alpha=0.3)
    scree_path = str((out_dir / "PCA_joint_scree.png").resolve())
    plt.tight_layout()
    plt.savefig(scree_path, dpi=200)
    plt.close()

    # Cumulative explained variance
    plt.figure(figsize=(7, 5))
    plt.plot(pcs, cum_evr, 's-', lw=2)
    plt.axhline(0.95, color='r', ls='--', lw=1, label='95%')
    plt.axhline(0.99, color='g', ls='--', lw=1, label='99%')
    plt.xlabel("Principal Component (index)")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Joint PCA — Cumulative Explained Variance (T + Td)")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    cum_path = str((out_dir / "PCA_joint_cumulative.png").resolve())
    plt.tight_layout()
    plt.savefig(cum_path, dpi=200)
    plt.close()

    return {"scree": scree_path, "cumulative": cum_path}


# (J2.8-alt) Function: plot_joint_pc_loadings_two_figs — 9 T-PCs and 9 Td-PCs in two clear figures
def plot_joint_pc_loadings_two_figs(
    out_dir: Path,
    W: np.ndarray,
    H_first21: np.ndarray,
    k_to_plot: int = 9,
    scaling_mode: str = 'zscore',
    invert_y: bool = False,
    color_T: str = 'crimson',
    color_Td: str = 'royalblue',
    lw: float = 2.0
) -> Dict[str, str]:
    """
    Generate two multi-panel figures (3×3 each) with no overlap:
      - Figure A: first k_to_plot temperature-part PC loadings (W_T[:, 0..k_to_plot-1])
      - Figure B: first k_to_plot dew-point-part PC loadings (W_Td[:, 0..k_to_plot-1])

    Inputs
    - out_dir: Output directory (Path).
    - W: Joint loadings (42 × p), PCs are columns; rows 0..20=T, 21..41=Td.
    - H_first21: Height array (length 21) in meters for the y-axis.
    - k_to_plot: Number of PCs to visualize per variable (default 9).
    - scaling_mode: String tag for figure titles ('none' | 'zscore' | 'block').
    - invert_y: If True, invert the y-axis (top=surface). Default False (bottom=surface).
    - color_T, color_Td: Line colors for T and Td plots.
    - lw: Line width.

    Returns
    - paths: dict with file paths to the saved figures:
        {'PCA_joint_T_PC1-9': ..., 'PCA_joint_Td_PC1-9': ...}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split W into temperature and dew-point parts
    p = W.shape[1]
    k_show = int(min(k_to_plot, p))
    W_T = W[:21, :k_show]
    W_Td = W[21:42, :k_show]

    # Compute symmetric x-limits per variable for visual comparability
    max_abs_T = float(np.max(np.abs(W_T))) if W_T.size > 0 else 1.0
    max_abs_Td = float(np.max(np.abs(W_Td))) if W_Td.size > 0 else 1.0
    xlim_T = (-1.05 * max_abs_T, 1.05 * max_abs_T)
    xlim_Td = (-1.05 * max_abs_Td, 1.05 * max_abs_Td)

    # Helper to make a 3×3 panel
    def make_panel(data_block: np.ndarray, var_name: str, color: str, xlim: tuple, fname: str) -> str:
        nrows, ncols = 3, 3
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(12, 10),
            constrained_layout=True,
            sharey=True
        )

        # Flatten axes for easy indexing
        axes_flat = axes.ravel()

        for i in range(nrows * ncols):
            ax = axes_flat[i]
            if i < data_block.shape[1]:
                ax.plot(data_block[:, i], H_first21, color=color, lw=lw)
                ax.axvline(0.0, color='k', lw=0.8, alpha=0.6)
                ax.set_title(f"PC{i+1}", fontsize=11, pad=3.5)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(xlim)
                # Only left column shows y-label to save space
                if i % ncols == 0:
                    ax.set_ylabel("Height (m)", fontsize=10)
                # Only bottom row shows x-label
                if i // ncols == (nrows - 1):
                    ax.set_xlabel(f"{var_name} loading", fontsize=10)
                # Set tick label sizes
                ax.tick_params(axis='both', which='major', labelsize=9)
                if invert_y:
                    ax.invert_yaxis()
            else:
                # Hide unused axes if k_show < 9
                ax.set_visible(False)

        fig.suptitle(
            f"Joint PCA — {var_name} loadings (PC1–PC{k_show})   [scaling: {scaling_mode}]",
            fontsize=13
        )
        out_path = str((out_dir / fname).resolve())
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    # Make both panels
    path_T = make_panel(
        data_block=W_T, var_name="Temperature (T)",
        color=color_T, xlim=xlim_T, fname=f"PCA_joint_T_PC1-{k_show}.png"
    )
    path_Td = make_panel(
        data_block=W_Td, var_name="Dew-point (Td)",
        color=color_Td, xlim=xlim_Td, fname=f"PCA_joint_Td_PC1-{k_show}.png"
    )

    return {"PCA_joint_T_PC1-9": path_T, "PCA_joint_Td_PC1-9": path_Td}


# (J2.9) Function: write_log_joint
def write_log_joint(
    out_dir: Path,
    step1_dir: Path,
    shapes: Dict[str, Tuple[int, ...]],
    scaling_info: Dict[str, object],
    ks_joint: Dict[float, int],
    saved_paths: Dict[str, str]
) -> str:
    """
    Write an inventory log for the Joint PCA step.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = str((out_dir / "Step2_JointPCA_inventory.log").resolve())
    lines = []
    lines.append("Ground-based Atmospheric Profiles — Step 2: Joint PCA (T + Td)")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("")
    lines.append("Inputs (from Step 1):")
    lines.append(f"  Step 1 directory: {str(step1_dir.resolve())}")
    lines.append("  - X_T_centered.csv (888 × 21): Centered temperature matrix (X_T − μ_T).")
    lines.append("  - X_Td_centered.csv (888 × 21): Centered dew-point matrix (X_Td − μ_Td).")
    lines.append("  - μ_T.csv (length 21), μ_Td.csv (length 21), H.csv (length 61).")
    lines.append("")
    lines.append("Core joint PCA relation:")
    lines.append("  - C = X_joint_centered · W_joint")
    lines.append("    where C: scores (n_samples × p),")
    lines.append("          X_joint_centered: (888 × 42) = [T(21) | Td(21)],")
    lines.append("          W_joint: loadings (42 × p).")
    lines.append("")
    lines.append("Shapes (as used):")
    for k, v in shapes.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Scaling applied:")
    for k, v in scaling_info.items():
        if isinstance(v, np.ndarray):
            lines.append(f"  - {k}: array shape {v.shape}")
        else:
            lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Selected number of components (k) by cumulative explained variance targets (joint):")
    for t, k in ks_joint.items():
        lines.append(f"  - {int(t*100)}% target: k = {k}")
    lines.append("")
    lines.append("Output files:")
    for key in sorted(saved_paths.keys()):
        lines.append(f"  - {key}: {saved_paths[key]}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - If scaling_mode='zscore', loadings are dimensionless; compare T vs Td shapes rather than amplitudes.")
    lines.append("  - If scaling_mode='none', loadings retain physical units and relative magnitudes.")
    lines.append("  - PC signs are arbitrary; interpret relative shapes and co-variations of T and Td.")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return log_path


# Main runner (callable from VSCode or notebooks)
def run_step2_joint(config: Step2JointConfig = None):
    if config is None:
        config = Step2JointConfig()

    step1_dir = config.step1_dir
    out_dir = config.out_dir_base / config.out_subfolder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Step 1
    data = load_step1_outputs(step1_dir)
    X_Tc = data['X_T_centered']    # (888 × 21)
    X_Tdc = data['X_Td_centered']  # (888 × 21)
    H = data['H']                  # (61,)
    H_first21 = H[:21]

    # Build joint matrix with chosen scaling
    joint = build_joint_matrix(X_Tc, X_Tdc, scaling_mode=config.scaling_mode)
    X_joint = joint['X_joint']                # (888 × 42)
    scaling_info = joint['scaling_info']
    feature_cols = joint['feature_cols']

    shapes = {
        "X_T_centered": X_Tc.shape,
        "X_Td_centered": X_Tdc.shape,
        "X_joint": X_joint.shape,
        "H": H.shape
    }

    # Run PCA (up to 42 components)
    pca_joint = run_pca(X_joint, n_components=X_joint.shape[1])
    ks_joint = decide_k(pca_joint['cum_explained_variance_ratio'], config.variance_targets)

    # Save outputs
    saved_paths = {}
    paths_joint = save_joint_outputs(
        out_dir=out_dir,
        W=pca_joint['W'],
        C=pca_joint['C'],
        ev=pca_joint['explained_variance'],
        evr=pca_joint['explained_variance_ratio'],
        cum_evr=pca_joint['cum_explained_variance_ratio'],
        ks=ks_joint,
        sample_index=X_Tc.index,
        H_first21=H_first21
    )
    saved_paths.update(paths_joint)

    # Determine how many PCs to show (up to 9)
    k_show = min(9, pca_joint['W'].shape[1])  # or min(9, ks_joint.get(0.99, 9))
    two_fig_paths = plot_joint_pc_loadings_two_figs(
        out_dir=out_dir,
        W=pca_joint['W'],
        H_first21=H_first21,
        k_to_plot=k_show,
        scaling_mode=config.scaling_mode,
        invert_y=False  # set True if you prefer height decreasing downward
    )
    saved_paths.update(two_fig_paths)

    # Log
    log_path = write_log_joint(out_dir, step1_dir, shapes, scaling_info, ks_joint, saved_paths)

    # Console summary
    print("Step 2 Joint PCA (T + Td) completed.")
    print(f"- Joint k (targets): {ks_joint}")
    print(f"Outputs saved in: {str(out_dir.resolve())}")
    print(f"Inventory log: {log_path}")

    return {
        "out_dir": str(out_dir.resolve()),
        "ks_joint": ks_joint,
        "log_path": log_path,
        "saved_paths": saved_paths,
        "feature_cols": feature_cols,
        "scaling_info": scaling_info
    }


# Entry point for VSCode "Run Python File"
if __name__ == "__main__":
    # Example: z-score to balance T and Td contributions and visualize 99% mode
    run_step2_joint(
        Step2JointConfig(
            step1_dir=Path("./Step1_Processed"),
            out_dir_base=Path(r"C:\Users\24573\OneDrive - The University of Hong Kong\PhD\Project\atmospheric sounding\prior-analysis"),
            variance_targets=[0.95, 0.99],
            scaling_mode='zscore',   # options: 'none' | 'zscore' | 'block'
            save_plots=True
        )
    )