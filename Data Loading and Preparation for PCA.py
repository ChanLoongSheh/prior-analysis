# Function 0: Main Script — Step 1: Data Loading and Preparation
# Author: (Your Name)
# Description:
#   This script implements Step 1 from the project proposal:
#   - Data loading and preparation for 888 atmospheric profiles.
#   - Reads MATLAB .mat files, extracts λ (lambda), H, T_atm, T_DP.
#   - Saves λ and H once; aggregates T_atm and T_DP (first 21 layers) across files.
#   - Builds matrices X_T (temperature) and X_Td (dewpoint) of shape (888, 21).
#   - Centers the matrices by subtracting their mean profiles: X_T_centered, X_Td_centered.
#   - Generates a detailed log (inventory) describing outputs and locations.
#
# Variables and meanings (readable font):
#   λ (lambda): Wavelength grid (vector), common to all files.
#   H: Height grid (vector), common to all files.
#   T_atm: Atmospheric temperature profile (vector of length 61 in each file).
#   T_DP: Dew-point temperature profile (vector of length 61 in each file).
#   X_T: Temperature matrix (888 × 21) built from T_atm (first 21 elements).
#   X_Td: Dew-point matrix (888 × 21) built from T_DP (first 21 elements).
#   μ_T (mu_T): Mean temperature profile (length 21).
#   μ_Td (mu_Td): Mean dew-point profile (length 21).
#   X_T_centered = X_T − μ_T (broadcast over rows).
#   X_Td_centered = X_Td − μ_Td (broadcast over rows).
#
# How to call:
#   - As a script: python step1_prepare_profiles.py
#   - As a module:
#       from step1_prepare_profiles import main
#       main()
#
# Notes:
#   - Requires: numpy, pandas, scipy (for scipy.io.loadmat). Optional: h5py (fallback for v7.3 MAT-files).
#   - All outputs are saved into a newly created folder in the current working directory.
#   - Row indices in the T_atm and T_DP CSVs equal the %d index parsed from file names.

import os
import re
import sys
import json
import time
import math
import shutil
import typing as t
from datetime import datetime

import numpy as np
import pandas as pd

# Try SciPy first for MAT loading; optionally fallback to h5py for MATLAB v7.3 files.
from scipy.io import loadmat
try:
    import h5py
    H5PY_AVAILABLE = True
except Exception:
    H5PY_AVAILABLE = False


# ------------------------------
# Function 1: list_and_index_mat_files()
# ------------------------------
def list_and_index_mat_files(root_dir: str) -> t.Dict[int, str]:
    """
    Purpose:
        List and index all MATLAB files in root_dir that match the naming pattern:
        'Rad_NewModel66layers_TDpP4Aerosol_HKObservatory0-100-5000m_6-1-15km_%d.mat'
        Extract the integer index %d, return a mapping {index: filepath}.

    Returns:
        files_by_idx: dict mapping file index (int) -> absolute filepath (str)
    """
    pattern = re.compile(
        r"^Rad_NewModel66layers_TDpP4Aerosol_HKObservatory0-100-5000m_6-1-15km_(\d+)\.mat$",
        re.IGNORECASE
    )
    files_by_idx: t.Dict[int, str] = {}

    for fname in os.listdir(root_dir):
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            files_by_idx[idx] = os.path.join(root_dir, fname)

    return dict(sorted(files_by_idx.items(), key=lambda kv: kv[0]))


# ------------------------------
# Function 2: load_one_mat()
# ------------------------------
def load_one_mat(mat_path: str) -> t.Dict[str, np.ndarray]:
    """
    Purpose:
        Load one .mat file and return variables as 1-D numpy arrays:
        - 'lambda' (λ)
        - 'H'
        - 'T_atm'
        - 'T_DP'

    Strategy:
        - First, try scipy.io.loadmat (for MATLAB <= v7.2).
        - If it fails due to v7.3 (HDF5-based), and h5py is available, use h5py.

    Returns:
        dict with keys: 'lambda', 'H', 'T_atm', 'T_DP' (each is 1-D np.ndarray)

    Raises:
        RuntimeError on failure or missing variables.
    """
    def _to_1d(a) -> np.ndarray:
        arr = np.array(a).squeeze()
        if arr.ndim != 1:
            raise RuntimeError(f"Expected a 1-D array, got shape {arr.shape}.")
        return arr.astype(float)

    # Attempt SciPy load
    try:
        mdict = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        # MATLAB saves may include extra keys starting with '__'
        if 'lambda' not in mdict or 'H' not in mdict or 'T_atm' not in mdict or 'T_DP' not in mdict:
            # Try a case-insensitive search
            key_map = {k.lower(): k for k in mdict.keys()}
            lam_key = key_map.get('lambda')
            H_key = key_map.get('h')
            Tatm_key = key_map.get('t_atm')
            Tdp_key = key_map.get('t_dp')
            if not all([lam_key, H_key, Tatm_key, Tdp_key]):
                raise KeyError("Missing one or more required keys in MAT file (lambda, H, T_atm, T_DP).")

            lam = _to_1d(mdict[lam_key])
            H = _to_1d(mdict[H_key])
            T_atm = _to_1d(mdict[Tatm_key])
            T_DP = _to_1d(mdict[Tdp_key])
        else:
            lam = _to_1d(mdict['lambda'])
            H = _to_1d(mdict['H'])
            T_atm = _to_1d(mdict['T_atm'])
            T_DP = _to_1d(mdict['T_DP'])

        return {'lambda': lam, 'H': H, 'T_atm': T_atm, 'T_DP': T_DP}

    except Exception as e_scipy:
        # Try h5py if available (MATLAB v7.3)
        if not H5PY_AVAILABLE:
            raise RuntimeError(f"Failed to load {mat_path} with scipy.io.loadmat and h5py not available. "
                               f"Error: {e_scipy}") from e_scipy

        try:
            with h5py.File(mat_path, 'r') as f:
                def read_var(name: str) -> np.ndarray:
                    # MAT v7.3 variables are datasets; we read and squeeze
                    if name in f:
                        data = np.array(f[name])
                    elif name.capitalize() in f:
                        data = np.array(f[name.capitalize()])
                    else:
                        # Attempt case-insensitive search
                        cands = [k for k in f.keys() if k.lower() == name.lower()]
                        if not cands:
                            raise KeyError(f"Variable {name} not found in {mat_path}.")
                        data = np.array(f[cands[0]])
                    return np.squeeze(data).astype(float)

                lam = read_var('lambda')
                H = read_var('H')
                T_atm = read_var('T_atm')
                T_DP = read_var('T_DP')

                # Ensure 1-D
                if lam.ndim != 1 or H.ndim != 1 or T_atm.ndim != 1 or T_DP.ndim != 1:
                    raise RuntimeError(f"Variables not 1-D after loading {mat_path}.")

                return {'lambda': lam, 'H': H, 'T_atm': T_atm, 'T_DP': T_DP}

        except Exception as e_h5:
            raise RuntimeError(f"Failed to load {mat_path} with h5py (v7.3). Error: {e_h5}") from e_h5


# ------------------------------
# Function 3: aggregate_profiles_and_save_csvs()
# ------------------------------
def aggregate_profiles_and_save_csvs(
    files_by_idx: t.Dict[int, str],
    out_dir: str,
    first_n_layers: int = 21,
    atol_same_grid: float = 1e-10
) -> t.Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, t.List[int], t.List[str]]:
    """
    Purpose:
        - Load all .mat files (indexed), validate λ and H consistency.
        - Extract first 'first_n_layers' (default 21) elements from T_atm and T_DP.
        - Build DataFrames with row index equal to file index (%d).
        - Save λ.csv, H.csv, T_atm_2km.csv, T_DP_2km.csv into out_dir.

    Returns:
        df_T_atm (DataFrame): shape (#files, first_n_layers), index = file indices
        df_T_DP  (DataFrame): shape (#files, first_n_layers), index = file indices
        lambda_ref (np.ndarray): reference λ vector saved
        H_ref (np.ndarray): reference H vector saved
        sorted_indices (List[int]): sorted list of file indices processed
        notes (List[str]): notes for the log about λ/H consistency, counts, etc.
    """
    notes: t.List[str] = []
    indices = list(files_by_idx.keys())
    if not indices:
        raise RuntimeError("No files found matching the required pattern.")

    sorted_indices = sorted(indices)

    lambda_ref = None
    H_ref = None
    lambda_mismatch_max = 0.0
    H_mismatch_max = 0.0

    # Preallocate lists
    T_atm_rows: t.List[np.ndarray] = []
    T_DP_rows: t.List[np.ndarray] = []

    for idx in sorted_indices:
        mat_path = files_by_idx[idx]
        data = load_one_mat(mat_path)
        lam = data['lambda']
        H = data['H']
        T_atm = data['T_atm']
        T_DP = data['T_DP']

        # Validate expected lengths
        if T_atm.size < first_n_layers or T_DP.size < first_n_layers:
            raise RuntimeError(
                f"File {mat_path} has insufficient vertical levels. "
                f"Expected >= {first_n_layers}, got T_atm={T_atm.size}, T_DP={T_DP.size}."
            )

        if lambda_ref is None:
            lambda_ref = lam.copy()
            H_ref = H.copy()
        else:
            # Check alignment with the first file
            if lam.shape != lambda_ref.shape:
                raise RuntimeError(f"λ (lambda) shape mismatch in {mat_path}: {lam.shape} vs {lambda_ref.shape}.")
            if H.shape != H_ref.shape:
                raise RuntimeError(f"H shape mismatch in {mat_path}: {H.shape} vs {H_ref.shape}.")

            lambda_mismatch_max = max(lambda_mismatch_max, float(np.max(np.abs(lam - lambda_ref))))
            H_mismatch_max = max(H_mismatch_max, float(np.max(np.abs(H - H_ref))))

        # Take first 21 (default) levels: represent the 0–2 km sampled points
        T_atm_rows.append(T_atm[:first_n_layers].astype(float))
        T_DP_rows.append(T_DP[:first_n_layers].astype(float))

    # Consistency notes
    notes.append(f"Number of files processed: {len(sorted_indices)}")
    notes.append(f"Max |Δλ| across files: {lambda_mismatch_max:.3e}")
    notes.append(f"Max |ΔH| across files: {H_mismatch_max:.3e}")
    if lambda_mismatch_max > atol_same_grid or H_mismatch_max > atol_same_grid:
        notes.append("WARNING: Detected non-negligible differences in λ or H across files.")
    else:
        notes.append("λ and H are consistent across files within tolerance.")

    # Build DataFrames with row index = file indices
    df_T_atm = pd.DataFrame(
        data=np.vstack(T_atm_rows),
        index=sorted_indices,
        columns=[f"level_{i}" for i in range(first_n_layers)]
    )
    df_T_atm.index.name = "file_index"

    df_T_DP = pd.DataFrame(
        data=np.vstack(T_DP_rows),
        index=sorted_indices,
        columns=[f"level_{i}" for i in range(first_n_layers)]
    )
    df_T_DP.index.name = "file_index"

    # Save outputs
    lambda_path = os.path.join(out_dir, "lambda.csv")
    H_path = os.path.join(out_dir, "H.csv")
    T_atm_path = os.path.join(out_dir, "T_atm_2km.csv")
    T_DP_path = os.path.join(out_dir, "T_DP_2km.csv")

    pd.DataFrame({"lambda": lambda_ref}).to_csv(lambda_path, index=False)
    pd.DataFrame({"H": H_ref}).to_csv(H_path, index=False)
    df_T_atm.to_csv(T_atm_path)
    df_T_DP.to_csv(T_DP_path)

    notes.append(f"Saved λ to: {lambda_path}")
    notes.append(f"Saved H to: {H_path}")
    notes.append(f"Saved T_atm (first {first_n_layers}) to: {T_atm_path}")
    notes.append(f"Saved T_DP  (first {first_n_layers}) to: {T_DP_path}")

    return df_T_atm, df_T_DP, lambda_ref, H_ref, sorted_indices, notes


# ------------------------------
# Function 4: prepare_centered_matrices()
# ------------------------------
def prepare_centered_matrices(
    df_T_atm: pd.DataFrame,
    df_T_DP: pd.DataFrame,
    out_dir: str
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, str, str]:
    """
    Purpose:
        From aggregated T_atm and T_DP DataFrames (shape: #files × 21):
        - Construct X_T and X_Td.
        - Compute mean profiles μ_T and μ_Td.
        - Center the matrices: X_T_centered = X_T − μ_T, X_Td_centered = X_Td − μ_Td.
        - Save X_T_centered.csv, X_Td_centered.csv, mu_T.csv, mu_Td.csv.

    Returns:
        X_T, X_Td, mu_T, mu_Td, paths of saved files (four strings)
    """
    # X_T and X_Td (raw)
    X_T = df_T_atm.to_numpy(dtype=float)
    X_Td = df_T_DP.to_numpy(dtype=float)

    # Mean profiles
    mu_T = X_T.mean(axis=0)
    mu_Td = X_Td.mean(axis=0)

    # Centering (broadcast subtraction)
    X_T_centered = X_T - mu_T
    X_Td_centered = X_Td - mu_Td

    # Save outputs
    X_Tc_path = os.path.join(out_dir, "X_T_centered.csv")
    X_Tdc_path = os.path.join(out_dir, "X_Td_centered.csv")
    mu_T_path = os.path.join(out_dir, "mu_T.csv")
    mu_Td_path = os.path.join(out_dir, "mu_Td.csv")

    # Keep the same row index as df_T_atm/df_T_DP for centered outputs
    df_X_Tc = pd.DataFrame(X_T_centered, index=df_T_atm.index, columns=df_T_atm.columns)
    df_X_Tdc = pd.DataFrame(X_Td_centered, index=df_T_DP.index, columns=df_T_DP.columns)
    df_mu_T = pd.DataFrame({"mu_T": mu_T})
    df_mu_Td = pd.DataFrame({"mu_Td": mu_Td})

    df_X_Tc.to_csv(X_Tc_path)
    df_X_Tdc.to_csv(X_Tdc_path)
    df_mu_T.to_csv(mu_T_path, index=False)
    df_mu_Td.to_csv(mu_Td_path, index=False)

    return X_T, X_Td, mu_T, mu_Td, X_Tc_path, X_Tdc_path, mu_T_path, mu_Td_path


# ------------------------------
# Function 5: write_inventory_log()
# ------------------------------
def write_inventory_log(
    out_dir: str,
    root_dir: str,
    df_T_atm: pd.DataFrame,
    df_T_DP: pd.DataFrame,
    lambda_ref: np.ndarray,
    H_ref: np.ndarray,
    sorted_indices: t.List[int],
    notes: t.List[str],
    X_T: np.ndarray,
    X_Td: np.ndarray,
    mu_T: np.ndarray,
    mu_Td: np.ndarray,
    saved_paths: t.Dict[str, str]
) -> str:
    """
    Purpose:
        Create a detailed log file summarizing:
        - What was processed.
        - The meaning of each output file and where it is saved.
        - Basic statistics (shapes, indices, etc.).

    Returns:
        path to the log file (str)
    """
    log_path = os.path.join(out_dir, "processing_inventory.log")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Ground-based Atmospheric Profiles — Step 1: Data Loading and Preparation\n")
        f.write(f"Timestamp: {now}\n\n")

        f.write("Source data:\n")
        f.write(f"  Root directory: {root_dir}\n")
        f.write(f"  Number of files processed: {len(sorted_indices)}\n")
        f.write(f"  File index range: {min(sorted_indices)} .. {max(sorted_indices)}\n\n")

        f.write("Consistency checks:\n")
        for n in notes:
            f.write(f"  - {n}\n")
        f.write("\n")

        f.write("Variables (readable fonts and meanings):\n")
        f.write("  - λ (lambda): Wavelength grid (vector), common to all files.\n")
        f.write("  - H: Height grid (vector), common to all files.\n")
        f.write("  - T_atm: Atmospheric temperature profile at multiple vertical levels (61 in raw files).\n")
        f.write("  - T_DP: Dew-point temperature profile at multiple vertical levels (61 in raw files).\n")
        f.write("  - X_T: Temperature matrix (888 × 21), constructed from T_atm first 21 levels (0–2 km).\n")
        f.write("  - X_Td: Dew-point matrix (888 × 21), constructed from T_DP first 21 levels (0–2 km).\n")
        f.write("  - μ_T (mu_T): Mean temperature profile (length 21) across the 888 samples.\n")
        f.write("  - μ_Td (mu_Td): Mean dew-point profile (length 21) across the 888 samples.\n")
        f.write("  - X_T_centered = X_T − μ_T (row-wise broadcast).\n")
        f.write("  - X_Td_centered = X_Td − μ_Td (row-wise broadcast).\n\n")

        f.write("Shapes and basic stats:\n")
        f.write(f"  - λ shape: {lambda_ref.shape}\n")
        f.write(f"  - H shape: {H_ref.shape}\n")
        f.write(f"  - T_atm (first 21) DataFrame shape: {df_T_atm.shape}\n")
        f.write(f"  - T_DP  (first 21) DataFrame shape: {df_T_DP.shape}\n")
        f.write(f"  - X_T shape: {X_T.shape}\n")
        f.write(f"  - X_Td shape: {X_Td.shape}\n\n")

        f.write("Output files and their meanings:\n")
        f.write(f"  - {saved_paths['lambda']}\n")
        f.write("      λ (lambda) wavelength grid saved once (common to all files).\n")
        f.write(f"  - {saved_paths['H']}\n")
        f.write("      H height grid saved once (common to all files).\n")
        f.write(f"  - {saved_paths['T_atm_2km']}\n")
        f.write("      Aggregated T_atm profiles (first 21 levels, 0–2 km). Rows indexed by file index %d.\n")
        f.write(f"  - {saved_paths['T_DP_2km']}\n")
        f.write("      Aggregated T_DP profiles (first 21 levels, 0–2 km). Rows indexed by file index %d.\n")
        f.write(f"  - {saved_paths['X_T_centered']}\n")
        f.write("      PCA-ready centered temperature matrix (X_T − μ_T). Index aligned to T_atm_2km.\n")
        f.write(f"  - {saved_paths['X_Td_centered']}\n")
        f.write("      PCA-ready centered dew-point matrix (X_Td − μ_Td). Index aligned to T_DP_2km.\n")
        f.write(f"  - {saved_paths['mu_T']}\n")
        f.write("      μ_T (mean temperature profile, length 21) for centering.\n")
        f.write(f"  - {saved_paths['mu_Td']}\n")
        f.write("      μ_Td (mean dew-point profile, length 21) for centering.\n\n")

        f.write("Notes for downstream steps:\n")
        f.write("  - The centered matrices are ready for PCA (Step 2): C = X_centered · W.\n")
        f.write("  - No detector system effects are included here (as requested for this stage).\n")
        f.write("  - Row indices in the CSVs correspond to the %d index parsed from file names.\n")

    return log_path


def main():
    # User-provided root folder containing 888 .mat files
    root = r"..\Data\888ground Radiance"

    # Create a new output directory in the current working directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.getcwd(), f"Step1_Processed_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # List and index files
    files_by_idx = list_and_index_mat_files(root)
    if len(files_by_idx) == 0:
        raise RuntimeError(f"No .mat files found in {root} matching the required naming pattern.")

    # Aggregate and save λ, H, T_atm_2km, T_DP_2km
    df_T_atm, df_T_DP, lambda_ref, H_ref, sorted_indices, notes = aggregate_profiles_and_save_csvs(
        files_by_idx=files_by_idx,
        out_dir=out_dir,
        first_n_layers=21,
        atol_same_grid=1e-10
    )

    # Prepare centered matrices and save them with mean profiles
    X_T, X_Td, mu_T, mu_Td, X_Tc_path, X_Tdc_path, mu_T_path, mu_Td_path = prepare_centered_matrices(
        df_T_atm=df_T_atm,
        df_T_DP=df_T_DP,
        out_dir=out_dir
    )

    # Compose saved paths dictionary for logging
    saved_paths = {
        "lambda": os.path.join(out_dir, "lambda.csv"),
        "H": os.path.join(out_dir, "H.csv"),
        "T_atm_2km": os.path.join(out_dir, "T_atm_2km.csv"),
        "T_DP_2km": os.path.join(out_dir, "T_DP_2km.csv"),
        "X_T_centered": X_Tc_path,
        "X_Td_centered": X_Tdc_path,
        "mu_T": mu_T_path,
        "mu_Td": mu_Td_path,
    }

    # Write detailed inventory log
    log_path = write_inventory_log(
        out_dir=out_dir,
        root_dir=root,
        df_T_atm=df_T_atm,
        df_T_DP=df_T_DP,
        lambda_ref=lambda_ref,
        H_ref=H_ref,
        sorted_indices=sorted_indices,
        notes=notes,
        X_T=X_T,
        X_Td=X_Td,
        mu_T=mu_T,
        mu_Td=mu_Td,
        saved_paths=saved_paths
    )

    print("Step 1 completed.")
    print(f"All outputs saved under: {out_dir}")
    print(f"Detailed log: {log_path}")


if __name__ == "__main__":
    main()