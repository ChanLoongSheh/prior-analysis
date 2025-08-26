# (2) Perform Principal Component Analysis (PCA) on centered profiles
# Title: Step 2 PCA Module for Ground-based Atmospheric Profiles (VSCode-ready)
# Author: [Your Name]
# Date: 2025-08-26
# Description:
#   Implements Step 2 from README.md without command-line arguments.
#   - Applies PCA to centered temperature and dew-point matrices (888 × 21).
#   - Produces principal components (W), scores (C), explained variance metrics.
#   - Selects k values by cumulative explained variance targets (e.g., 95%, 99%).
#   - Saves all outputs to a new timestamped folder and writes a detailed log file.
#
# Key variables (readable fonts and meanings):
#   - X_centered: Centered data matrix (888 × 21). For T: X_T_centered; for Td: X_Td_centered.
#   - W: Principal component matrix (21 × p), columns are EOFs in height space.
#   - C: Scores (888 × p) = X_centered · W, coordinates of each sample in PC space.
#   - explained_variance: Eigenvalue-like quantities per PC (sklearn definition).
#   - explained_variance_ratio: Fraction of variance explained by each PC.
#   - cumulative_explained_variance_ratio: Cumulative fraction of explained variance.
#   - k95, k99: Minimal number of PCs to reach 95% and 99% cumulative explained variance.
#
# Notes:
#   - This script only uses Step 1 outputs. No detector system effects are involved at this stage.
#   - It expects the following Step 1 CSV files:
#       X_T_centered.csv, X_Td_centered.csv, mu_T.csv, mu_Td.csv, H.csv
#   - Outputs are saved into a new folder (timestamped by default).

from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# =========================
# Configuration (edit here)
# =========================

@dataclass
class Step2Config:
    # Path to Step 1 processed outputs (from processing_inventory.log)
    step1_dir: Path = Path("./Step1_Processed")
    # Base output directory for Step 2; a timestamped subfolder will be created inside this base
    out_dir_base: Path = Path(r"C:\Users\24573\OneDrive - The University of Hong Kong\PhD\Project\atmospheric sounding\prior-analysis")
    # Targets for cumulative explained variance
    variance_targets: List[float] = None
    # Whether to save diagnostic plots
    save_plots: bool = True
    # Name of the subfolder to create (auto-timestamped below if None)
    out_subfolder_name: str = None

    def __post_init__(self):
        if self.variance_targets is None:
            self.variance_targets = [0.95, 0.99]
        if self.out_subfolder_name is None:
            ts = datetime.datetime.now().strftime("Step2_PCA_%Y%m%d_%H%M%S")
            self.out_subfolder_name = ts


# (2.1) Function: read_vector_csv_robust
def read_vector_csv_robust(path: Path, expected_len: int) -> np.ndarray:
    """
    Robustly read a 1D vector from CSV that may be saved as:
      - a column vector (n×1) with/without header and with/without index
      - a row vector (1×n) with/without header and with/without index

    Strategy:
      1) Try reading with index_col=0. If that removes all data columns, fall back.
      2) Collect all numeric cells and flatten; select a length-matching interpretation.
      3) If a clear (n×1) or (1×n) shape matches expected_len, use it.
      4) As last resort, take the first expected_len numeric values after flattening.

    Parameters
    ----------
    path : Path
        CSV file path.
    expected_len : int
        Expected vector length (e.g., 21 for μ_T, 61 for H).

    Returns
    -------
    vec : np.ndarray of shape (expected_len,)
    """
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")

    def numeric_flat(df: pd.DataFrame) -> np.ndarray:
        df_num = df.apply(pd.to_numeric, errors='coerce')
        arr = df_num.values.ravel()
        return arr[~np.isnan(arr)]

    # Attempt 1: with index_col=0
    try:
        df1 = pd.read_csv(path, index_col=0)
        if df1.shape[1] == 0:
            # No data columns left; re-read without index
            df1 = pd.read_csv(path)
    except Exception:
        df1 = pd.read_csv(path)

    # If df1 has a clear single row or single column of size expected_len, use it
    if df1.shape[0] == expected_len and df1.shape[1] >= 1:
        col0 = pd.to_numeric(df1.iloc[:, 0], errors='coerce').values
        if np.count_nonzero(~np.isnan(col0)) == expected_len:
            return col0.astype(float)

    if df1.shape[1] == expected_len and df1.shape[0] >= 1:
        row0 = pd.to_numeric(df1.iloc[0, :], errors='coerce').values
        if np.count_nonzero(~np.isnan(row0)) == expected_len:
            return row0.astype(float)

    # Fallback: flatten numeric content and pick the best interpretation
    flat1 = numeric_flat(df1)
    if flat1.size >= expected_len:
        return flat1[:expected_len].astype(float)

    # Attempt 2: read with header=None (sometimes needed)
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


# (2.2) Function: load_step1_outputs
def load_step1_outputs(step1_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load required Step 1 outputs from CSV.

    Parameters
    ----------
    step1_dir : Path
        Path to the Step1_Processed directory.

    Returns
    -------
    data : dict
        Dictionary with keys:
          - 'X_T_centered' : DataFrame (888 × 21)
          - 'X_Td_centered': DataFrame (888 × 21)
          - 'mu_T'         : np.ndarray (length 21)
          - 'mu_Td'        : np.ndarray (length 21)
          - 'H'            : np.ndarray (length 61), height grid
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

    # Matrices: keep row/column indices as in Step 1
    X_T_centered = pd.read_csv(files['X_T_centered'], index_col=0)
    X_Td_centered = pd.read_csv(files['X_Td_centered'], index_col=0)

    # Vectors: use robust reader (fixes the bug you encountered)
    mu_T = read_vector_csv_robust(files['mu_T'], expected_len=21)
    mu_Td = read_vector_csv_robust(files['mu_Td'], expected_len=21)
    H = read_vector_csv_robust(files['H'], expected_len=61)

    data = {
        'X_T_centered': X_T_centered.apply(pd.to_numeric, errors='coerce'),
        'X_Td_centered': X_Td_centered.apply(pd.to_numeric, errors='coerce'),
        'mu_T': mu_T,
        'mu_Td': mu_Td,
        'H': H,
    }
    return data


# (2.3) Function: run_pca
def run_pca(X_centered: np.ndarray, n_components: int = None) -> Dict[str, np.ndarray]:
    """
    Run PCA on a centered matrix X_centered (n_samples × n_features).

    Parameters
    ----------
    X_centered : np.ndarray
        Centered data matrix.
    n_components : int or None
        Number of components to retain. If None, retain all features.

    Returns
    -------
    result : dict
        - 'W' : (n_features × p) principal component loadings (columns are PCs)
        - 'C' : (n_samples × p) scores
        - 'explained_variance' : (p,)
        - 'explained_variance_ratio' : (p,)
        - 'cum_explained_variance_ratio' : (p,)
        - 'components_raw' : (p × n_features) sklearn components_ (PCs as rows)
    """
    n_samples, n_features = X_centered.shape
    if n_components is None:
        n_components = n_features

    pca = PCA(n_components=n_components, svd_solver='full')
    C = pca.fit_transform(X_centered)  # scores (n_samples × p)
    components_raw = pca.components_   # (p × n_features), PCs as rows
    W = components_raw.T               # (n_features × p), PCs as columns
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


# (2.4) Function: decide_k
def decide_k(cum_evr: np.ndarray, targets: List[float]) -> Dict[float, int]:
    """
    Decide minimal number of components to reach each cumulative explained variance target.

    Parameters
    ----------
    cum_evr : np.ndarray
        Cumulative explained variance ratio (length p).
    targets : list of float
        Target cumulative explained variance levels (e.g., [0.95, 0.99]).

    Returns
    -------
    ks : dict
        Mapping {target: k_target}
    """
    ks = {}
    for t in targets:
        idx = np.searchsorted(cum_evr, t, side='left')
        k = int(idx + 1) if idx < len(cum_evr) else len(cum_evr)
        ks[float(t)] = k
    return ks


# (2.5) Function: save_pca_outputs
def save_pca_outputs(
    out_dir: Path,
    tag: str,
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
    Save PCA outputs to CSV files and return a dictionary of file paths.

    Parameters
    ----------
    out_dir : Path
        Output directory (will be created if not exists).
    tag : str
        'T' for temperature or 'Td' for dew-point.
    W : np.ndarray
        Loadings matrix (21 × p).
    C : np.ndarray
        Scores matrix (888 × p).
    ev, evr, cum_evr : np.ndarray
        Explained variance, ratio, and cumulative ratio.
    ks : dict
        Selected k values for targets.
    sample_index : pd.Index
        Row labels (length 888), passed through for scores.
    H_first21 : np.ndarray
        First 21 heights (0–2 km), used as the feature row index.

    Returns
    -------
    paths : dict
        Mapping from logical names to file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    p = W.shape[1]
    pc_cols = [f"PC{i+1}" for i in range(p)]

    # Loadings indexed by height (0–2 km)
    df_W = pd.DataFrame(W, index=H_first21, columns=pc_cols)
    df_W.index.name = "H_first21"

    # Scores keep sample indices
    df_C = pd.DataFrame(C, index=sample_index, columns=pc_cols)

    df_ev = pd.DataFrame({"explained_variance": ev}, index=pc_cols)
    df_evr = pd.DataFrame({"explained_variance_ratio": evr}, index=pc_cols)
    df_cumevr = pd.DataFrame({"cumulative_explained_variance_ratio": cum_evr}, index=pc_cols)

    paths = {}
    paths[f"PCA_{tag}_W_csv"] = str((out_dir / f"PCA_{tag}_W.csv").resolve())
    paths[f"PCA_{tag}_scores_csv"] = str((out_dir / f"PCA_{tag}_scores.csv").resolve())
    paths[f"PCA_{tag}_explained_variance_csv"] = str((out_dir / f"PCA_{tag}_explained_variance.csv").resolve())
    paths[f"PCA_{tag}_explained_variance_ratio_csv"] = str((out_dir / f"PCA_{tag}_explained_variance_ratio.csv").resolve())
    paths[f"PCA_{tag}_cumulative_explained_variance_ratio_csv"] = str((out_dir / f"PCA_{tag}_cumulative_explained_variance_ratio.csv").resolve())
    df_W.to_csv(paths[f"PCA_{tag}_W_csv"])
    df_C.to_csv(paths[f"PCA_{tag}_scores_csv"])
    df_ev.to_csv(paths[f"PCA_{tag}_explained_variance_csv"])
    df_evr.to_csv(paths[f"PCA_{tag}_explained_variance_ratio_csv"])
    df_cumevr.to_csv(paths[f"PCA_{tag}_cumulative_explained_variance_ratio_csv"])

    # Save reduced versions for selected k per target
    for t, k in ks.items():
        df_W_k = df_W.iloc[:, :k]
        df_C_k = df_C.iloc[:, :k]
        paths[f"PCA_{tag}_W_k_{int(t*100)}_csv"] = str((out_dir / f"PCA_{tag}_W_k{int(t*100)}.csv").resolve())
        paths[f"PCA_{tag}_scores_k_{int(t*100)}_csv"] = str((out_dir / f"PCA_{tag}_scores_k{int(t*100)}.csv").resolve())
        df_W_k.to_csv(paths[f"PCA_{tag}_W_k_{int(t*100)}_csv"])
        df_C_k.to_csv(paths[f"PCA_{tag}_scores_k_{int(t*100)}_csv"])

    return paths


# (2.6) Function: save_plots
def save_plots(out_dir: Path, tag: str, evr: np.ndarray, cum_evr: np.ndarray) -> Dict[str, str]:
    """
    Save scree plot and cumulative explained variance plot.

    Parameters
    ----------
    out_dir : Path
        Output directory.
    tag : str
        'T' or 'Td'.
    evr : np.ndarray
        Explained variance ratio.
    cum_evr : np.ndarray
        Cumulative explained variance ratio.

    Returns
    -------
    paths : dict
        Mapping to saved plot files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pcs = np.arange(1, len(evr) + 1)

    # Scree plot
    plt.figure(figsize=(7, 5))
    plt.plot(pcs, evr, 'o-', lw=2)
    plt.xlabel("Principal Component (index)")
    plt.ylabel("Explained Variance Ratio")
    plt.title(f"Scree Plot — {tag}")
    plt.grid(True, alpha=0.3)
    scree_path = str((out_dir / f"PCA_{tag}_scree.png").resolve())
    plt.tight_layout()
    plt.savefig(scree_path, dpi=200)
    plt.close()

    # Cumulative explained variance plot
    plt.figure(figsize=(7, 5))
    plt.plot(pcs, cum_evr, 's-', lw=2)
    plt.axhline(0.95, color='r', ls='--', lw=1, label='95%')
    plt.axhline(0.99, color='g', ls='--', lw=1, label='99%')
    plt.xlabel("Principal Component (index)")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title(f"Cumulative Explained Variance — {tag}")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    cum_path = str((out_dir / f"PCA_{tag}_cumulative.png").resolve())
    plt.tight_layout()
    plt.savefig(cum_path, dpi=200)
    plt.close()

    return {"scree": scree_path, "cumulative": cum_path}


# (2.7) Function: write_log
def write_log(
    out_dir: Path,
    step1_dir: Path,
    shapes: Dict[str, Tuple[int, ...]],
    ks_T: Dict[float, int],
    ks_Td: Dict[float, int],
    saved_paths: Dict[str, str]
) -> str:
    """
    Write a detailed log file of what was produced and where it was saved.

    Parameters
    ----------
    out_dir : Path
        Step 2 output directory.
    step1_dir : Path
        Step 1 directory (for provenance).
    shapes : dict
        Shapes of key arrays.
    ks_T, ks_Td : dict
        Selected k per target for T and Td.
    saved_paths : dict
        Map of logical names to file paths.

    Returns
    -------
    log_path : str
        Path to the log file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = str((out_dir / "Step2_PCA_inventory.log").resolve())
    lines = []
    lines.append("Ground-based Atmospheric Profiles — Step 2: PCA")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("")
    lines.append("Inputs (from Step 1):")
    lines.append(f"  Step 1 directory: {str(step1_dir.resolve())}")
    lines.append("  - X_T_centered.csv (888 × 21): Centered temperature matrix (X_T − μ_T).")
    lines.append("  - X_Td_centered.csv (888 × 21): Centered dew-point matrix (X_Td − μ_Td).")
    lines.append("  - μ_T.csv (length 21): Mean temperature profile used for centering.")
    lines.append("  - μ_Td.csv (length 21): Mean dew-point profile used for centering.")
    lines.append("  - H.csv (length 61): Height grid (first 21 levels correspond to 0–2 km).")
    lines.append("")
    lines.append("Core PCA relation:")
    lines.append("  - C = X_centered · W")
    lines.append("    where C: scores (coefficients) matrix (n_samples × p),")
    lines.append("          X_centered: centered data matrix (888 × 21),")
    lines.append("          W: principal component loadings (21 × p).")
    lines.append("")
    lines.append("Shapes (as used):")
    for k, v in shapes.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Selected number of components (k) by cumulative explained variance targets:")
    lines.append("  Temperature (T):")
    for t, k in ks_T.items():
        lines.append(f"    - {int(t*100)}% target: k = {k}")
    lines.append("  Dew-point (Td):")
    for t, k in ks_Td.items():
        lines.append(f"    - {int(t*100)}% target: k = {k}")
    lines.append("")
    lines.append("Output files and meanings:")
    descriptions = {
        "W": "Principal component loadings (21 × p). Rows indexed by first 21 heights (0–2 km). Columns PC1..PCp.",
        "scores": "Scores (coefficients) C = X_centered · W (888 × p). Rows indexed by sample (file index). Columns PC1..PCp.",
        "explained_variance": "Eigenvalue-like quantities per PC (sklearn definition).",
        "explained_variance_ratio": "Fraction of variance explained by each PC.",
        "cumulative_explained_variance_ratio": "Cumulative sum of explained variance ratio.",
        "W_k": "First k principal component loadings for the specified target (95% or 99%).",
        "scores_k": "Scores truncated to the first k PCs for the specified target (95% or 99%).",
        "plots": "Diagnostic plots: scree and cumulative explained variance."
    }
    for key in sorted(saved_paths.keys()):
        path = saved_paths[key]
        if "_W_k" in key:
            meaning_key = "W_k"
        elif "_scores_k" in key:
            meaning_key = "scores_k"
        elif key.endswith("_W_csv"):
            meaning_key = "W"
        elif key.endswith("_scores_csv"):
            meaning_key = "scores"
        elif key.endswith("explained_variance_csv"):
            meaning_key = "explained_variance"
        elif key.endswith("explained_variance_ratio_csv"):
            meaning_key = "explained_variance_ratio"
        elif key.endswith("cumulative_explained_variance_ratio_csv"):
            meaning_key = "cumulative_explained_variance_ratio"
        elif key.endswith("_scree.png") or key.endswith("_cumulative.png"):
            meaning_key = "plots"
        else:
            meaning_key = "N/A"

        lines.append(f"  - {path}")
        lines.append(f"      Meaning: [{meaning_key}] {descriptions.get(meaning_key, 'N/A')}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - No detector system effects, thermal emission, or noise considerations are included in this stage.")
    lines.append("  - Row indices are preserved from Step 1 where applicable.")
    lines.append("  - These outputs are ready for downstream statistical analysis (Step 3) and representative profile selection (Step 4).")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return log_path


# Main runner (callable from VSCode or notebooks)
def run_step2(config: Step2Config = None):
    if config is None:
        config = Step2Config()

    step1_dir = config.step1_dir
    out_dir = config.out_dir_base / config.out_subfolder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Step 1 outputs
    data = load_step1_outputs(step1_dir)
    X_Tc = data['X_T_centered']
    X_Tdc = data['X_Td_centered']
    mu_T = data['mu_T']    # kept for provenance
    mu_Td = data['mu_Td']
    H = data['H']          # length 61
    H_first21 = H[:21]

    # Basic checks
    if X_Tc.shape[1] != 21 or X_Tdc.shape[1] != 21:
        raise ValueError("Expected 21 features (first 21 levels, 0–2 km). Check Step 1 outputs.")
    shapes = {
        "X_T_centered": X_Tc.shape,
        "X_Td_centered": X_Tdc.shape,
        "mu_T": mu_T.shape,
        "mu_Td": mu_Td.shape,
        "H": H.shape,
    }

    # Run PCA for Temperature
    pca_T = run_pca(X_Tc.values, n_components=21)
    ks_T = decide_k(pca_T['cum_explained_variance_ratio'], config.variance_targets)

    # Run PCA for Dew-point
    pca_Td = run_pca(X_Tdc.values, n_components=21)
    ks_Td = decide_k(pca_Td['cum_explained_variance_ratio'], config.variance_targets)

    # Save outputs
    saved_paths = {}

    # Temperature outputs
    paths_T = save_pca_outputs(
        out_dir=out_dir,
        tag="T",
        W=pca_T['W'],
        C=pca_T['C'],
        ev=pca_T['explained_variance'],
        evr=pca_T['explained_variance_ratio'],
        cum_evr=pca_T['cum_explained_variance_ratio'],
        ks=ks_T,
        sample_index=X_Tc.index,
        H_first21=H_first21
    )
    saved_paths.update(paths_T)

    # Dew-point outputs
    paths_Td = save_pca_outputs(
        out_dir=out_dir,
        tag="Td",
        W=pca_Td['W'],
        C=pca_Td['C'],
        ev=pca_Td['explained_variance'],
        evr=pca_Td['explained_variance_ratio'],
        cum_evr=pca_Td['cum_explained_variance_ratio'],
        ks=ks_Td,
        sample_index=X_Tdc.index,
        H_first21=H_first21
    )
    saved_paths.update(paths_Td)

    # Optional plots
    if config.save_plots:
        plot_paths_T = save_plots(out_dir, "T", pca_T['explained_variance_ratio'], pca_T['cum_explained_variance_ratio'])
        plot_paths_Td = save_plots(out_dir, "Td", pca_Td['explained_variance_ratio'], pca_Td['cum_explained_variance_ratio'])
        saved_paths.update({f"PCA_T_{k}": v for k, v in plot_paths_T.items()})
        saved_paths.update({f"PCA_Td_{k}": v for k, v in plot_paths_Td.items()})

    # Write log
    log_path = write_log(out_dir, step1_dir, shapes, ks_T, ks_Td, saved_paths)

    # Console summary (visible in VSCode terminal/output)
    print("Step 2 PCA completed.")
    print(f"- Temperature k (targets): {ks_T}")
    print(f"- Dew-point   k (targets): {ks_Td}")
    print(f"Outputs saved in: {str(out_dir.resolve())}")
    print(f"Inventory log: {log_path}")

    return {
        "out_dir": str(out_dir.resolve()),
        "ks_T": ks_T,
        "ks_Td": ks_Td,
        "log_path": log_path,
        "saved_paths": saved_paths
    }


# Entry point for VSCode "Run Python File"
if __name__ == "__main__":
    # Edit CONFIG above if needed, then press Run in VSCode.
    run_step2()