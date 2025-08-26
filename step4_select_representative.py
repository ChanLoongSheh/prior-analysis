# Functionality: Step 4 — Select a Representative Atmospheric Profile
# Author: (Your Name)
# Context: Prior-Analysis Project — Redefining Atmospheric Weighting Functions using PCA (Step 4 only)
# Guarantees:
#   - Uses Step 1 outputs only.
#   - No command-line parsing.
#   - Saves all outputs to a new folder named without timestamp.
#   - Generates a detailed log describing what and where the outputs are.
#   - CSVs include headers and row indices.

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any

# ----------------------------
# Configuration (edit if needed)
# ----------------------------
# Input directory from Step 1 (as recorded in processing_inventory.log)
INPUT_DIR_STEP1 = Path("./Step1_Processed")

# Output directory for Step 4 (without timestamp)
output_parent = INPUT_DIR_STEP1.parent
OUTPUT_DIR_STEP4 = output_parent / "Step4_RepresentativeProfile"

# Filenames expected from Step 1
FN_X_T_CENTERED   = "X_T_centered.csv"   # X_T_centered = X_T − μ_T
FN_X_TD_CENTERED  = "X_Td_centered.csv"  # X_Td_centered = X_Td − μ_Td
FN_T_ATM_2KM      = "T_atm_2km.csv"      # X_T (0–2 km)
FN_T_DP_2KM       = "T_DP_2km.csv"       # X_Td (0–2 km)
FN_MU_T           = "mu_T.csv"           # μ_T (mean temperature profile, length 21)
FN_MU_TD          = "mu_Td.csv"          # μ_Td (mean dew-point profile, length 21)
FN_H              = "H.csv"              # Full height grid (we will take first 21 levels)

# =============================================================================
# Function 0 — Robust CSV utilities (hardened readers for Step 1 outputs)
# =============================================================================
def _extract_numeric_1d_from_df(df: pd.DataFrame) -> np.ndarray:
    """
    Flatten all numeric-like values from a DataFrame into a 1D numpy array.
    This is a last-resort extractor for μ vectors or H when layout is non-standard.
    """
    cols = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        cols.append(s)
    if len(cols) == 0:
        return np.array([], dtype=float)
    stacked = pd.concat(cols, axis=0)
    vals = stacked.dropna().astype(float).to_numpy()
    return vals

def read_vector_csv(path: Path, expected_len: int, name_hint: str = "") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Robustly read a 1D vector CSV (e.g., μ_T, μ_Td, H).
    Returns:
      - vec: numpy array of length 'expected_len'
      - meta: dict with 'source_layout' and diagnostics for logging
    Tries several patterns:
      1) Read with index_col=0; if single row -> take row; if single col -> take col.
      2) If no columns, re-read without index_col and try heuristics.
      3) As a last resort, flatten all numeric values and take the first expected_len entries.
    """
    meta = {"file": str(path), "attempts": []}

    # Attempt A: index_col=0, header=0
    try:
        df = pd.read_csv(path, index_col=0)
        meta["attempts"].append({"mode": "index_col=0", "shape": df.shape, "columns": list(df.columns)})
        # Single-row vector
        if df.shape[0] == 1 and df.shape[1] >= expected_len:
            vec = df.iloc[0, :expected_len].to_numpy(dtype=float)
            meta["source_layout"] = f"single-row with headers ({name_hint})"
            return vec, meta
        # Single-column vector
        if df.shape[1] == 1 and df.shape[0] >= expected_len:
            vec = df.iloc[:expected_len, 0].to_numpy(dtype=float)
            meta["source_layout"] = f"single-column with headers ({name_hint})"
            return vec, meta
        # Exact match either orientation
        if df.shape[1] == expected_len and df.shape[0] == 1:
            vec = df.iloc[0].to_numpy(dtype=float)
            meta["source_layout"] = f"1×{expected_len} row ({name_hint})"
            return vec, meta
        if df.shape[0] == expected_len and df.shape[1] == 1:
            vec = df.iloc[:, 0].to_numpy(dtype=float)
            meta["source_layout"] = f"{expected_len}×1 column ({name_hint})"
            return vec, meta
    except Exception as e:
        meta["attempts"].append({"mode": "index_col=0", "error": repr(e)})

    # Attempt B: read without index_col
    try:
        df2 = pd.read_csv(path)
        meta["attempts"].append({"mode": "no index_col", "shape": df2.shape, "columns": list(df2.columns)})

        # Drop a common artifact index column if present
        if "Unnamed: 0" in df2.columns:
            df2 = df2.drop(columns=["Unnamed: 0"])
            meta["attempts"][-1]["columns_after_drop"] = list(df2.columns)

        # Single-row
        if df2.shape[0] == 1 and df2.shape[1] >= expected_len:
            vec = df2.iloc[0, :expected_len].to_numpy(dtype=float)
            meta["source_layout"] = f"single-row no index ({name_hint})"
            return vec, meta
        # Single-column
        if df2.shape[1] == 1 and df2.shape[0] >= expected_len:
            vec = df2.iloc[:expected_len, 0].to_numpy(dtype=float)
            meta["source_layout"] = f"single-column no index ({name_hint})"
            return vec, meta
        # Exact match either orientation
        if df2.shape[1] == expected_len and df2.shape[0] == 1:
            vec = df2.iloc[0].to_numpy(dtype=float)
            meta["source_layout"] = f"1×{expected_len} row no index ({name_hint})"
            return vec, meta
        if df2.shape[0] == expected_len and df2.shape[1] == 1:
            vec = df2.iloc[:, 0].to_numpy(dtype=float)
            meta["source_layout"] = f"{expected_len}×1 column no index ({name_hint})"
            return vec, meta

        # If a named column matches the name hint directly, try it
        if name_hint and name_hint in df2.columns and df2[name_hint].shape[0] >= expected_len:
            vec = pd.to_numeric(df2[name_hint], errors="coerce").dropna().to_numpy(dtype=float)[:expected_len]
            meta["source_layout"] = f"column named '{name_hint}'"
            return vec, meta

    except Exception as e:
        meta["attempts"].append({"mode": "no index_col", "error": repr(e)})

    # Attempt C: last resort — flatten numeric content and take the first expected_len values
    try:
        # Try index_col=0 first; if it fails, load raw then flatten
        try:
            df_raw = pd.read_csv(path, index_col=0)
        except Exception:
            df_raw = pd.read_csv(path)
        vals = _extract_numeric_1d_from_df(df_raw)
        if vals.size >= expected_len:
            vec = vals[:expected_len]
            meta["source_layout"] = f"flattened numeric extraction ({name_hint})"
            return vec, meta
        else:
            # Try reading with no index and flatten again
            df_raw2 = pd.read_csv(path)
            vals2 = _extract_numeric_1d_from_df(df_raw2)
            if vals2.size >= expected_len:
                vec = vals2[:expected_len]
                meta["source_layout"] = f"flattened numeric extraction (no index) ({name_hint})"
                return vec, meta
            else:
                meta["attempts"].append({"mode": "flatten", "error": f"Only {vals2.size} numeric values found; need {expected_len}."})
    except Exception as e:
        meta["attempts"].append({"mode": "flatten", "error": repr(e)})

    # If we arrive here, we failed
    raise ValueError(f"Failed to read vector '{name_hint}' from {path}. Attempts: {meta['attempts']}")

def read_matrix_with_index(path: Path, expect_cols: int | None = None) -> pd.DataFrame:
    """
    Read a 2D matrix CSV with a row index and column headers, robust to common patterns.
    Returns a DataFrame with the original row index preserved.
    """
    # First try with index_col=0 (most Step 1 outputs comply)
    try:
        df = pd.read_csv(path, index_col=0)
        if expect_cols is None or df.shape[1] == expect_cols:
            return df
    except Exception:
        pass

    # Fallback: read without index_col, then set an index if an artifact exists
    df2 = pd.read_csv(path)
    if "Unnamed: 0" in df2.columns:
        df2 = df2.set_index("Unnamed: 0")
        df2.index.name = None
    else:
        # If the first column looks like an index (unique integer-like identifiers), set it as index
        first_col = df2.columns[0]
        fc = df2[first_col]
        try:
            fc_int = pd.to_numeric(fc, errors="coerce")
            looks_like_index = fc_int.notna().all() and (fc_int.astype(int) == fc_int).all()
        except Exception:
            looks_like_index = False
        if looks_like_index and df2.shape[1] > 1:
            df2 = df2.set_index(first_col)
            df2.index.name = None

    if expect_cols is not None and df2.shape[1] != expect_cols:
        raise ValueError(f"{path.name}: expected {expect_cols} columns, found {df2.shape[1]}.")
    return df2

# =============================================================================
# Function 1 — Load Step 1 data
# =============================================================================
def load_step1_data(input_dir: Path) -> dict:
    """
    Load the Step 1 outputs necessary for Step 4.

    Returns:
      - df_XT_centered: DataFrame (888 × 21)
      - df_XTd_centered: DataFrame (888 × 21)
      - df_T_atm: DataFrame (888 × 21)
      - df_T_dp: DataFrame (888 × 21)
      - mu_T: numpy array (length 21)
      - mu_Td: numpy array (length 21)
      - H_2km: numpy array (length 21)
      - parse_meta: diagnostics dict for logging (how μ and H were parsed)
    """
    # Load matrices with robust index handling
    df_XT_centered = read_matrix_with_index(input_dir / FN_X_T_CENTERED, expect_cols=21)
    df_XTd_centered = read_matrix_with_index(input_dir / FN_X_TD_CENTERED, expect_cols=21)
    df_T_atm = read_matrix_with_index(input_dir / FN_T_ATM_2KM, expect_cols=21)
    df_T_dp = read_matrix_with_index(input_dir / FN_T_DP_2KM, expect_cols=21)

    # Cross-check indices
    if not df_XT_centered.index.equals(df_XTd_centered.index):
        raise ValueError("Row indices of X_T_centered and X_Td_centered do not match.")
    if not df_T_atm.index.equals(df_T_dp.index):
        raise ValueError("Row indices of T_atm_2km and T_DP_2km do not match.")
    if not df_XT_centered.index.equals(df_T_atm.index):
        raise ValueError("Row indices of centered matrices and original profiles do not align.")

    # Read μ_T and μ_Td robustly as 1D vectors (length 21)
    mu_T, meta_mu_T = read_vector_csv(input_dir / FN_MU_T, expected_len=21, name_hint="μ_T")
    mu_Td, meta_mu_Td = read_vector_csv(input_dir / FN_MU_TD, expected_len=21, name_hint="μ_Td")

    # Read H (full) and slice first 21 levels (0–2 km)
    H_full, meta_H = read_vector_csv(input_dir / FN_H, expected_len=61, name_hint="H_full")
    H_2km = H_full[:21].copy()

    # Final shape checks
    if df_XT_centered.shape[1] != 21 or df_XTd_centered.shape[1] != 21:
        raise ValueError("Centered matrices must have exactly 21 columns (0–2 km).")
    if mu_T.shape[0] != 21 or mu_Td.shape[0] != 21:
        raise ValueError("Mean profiles μ_T and μ_Td must have length 21.")
    if H_2km.shape[0] != 21:
        raise ValueError("Height grid must provide at least the first 21 levels.")

    parse_meta = {"mu_T": meta_mu_T, "mu_Td": meta_mu_Td, "H": meta_H}
    return {
        "df_XT_centered": df_XT_centered,
        "df_XTd_centered": df_XTd_centered,
        "df_T_atm": df_T_atm,
        "df_T_dp": df_T_dp,
        "mu_T": mu_T,
        "mu_Td": mu_Td,
        "H_2km": H_2km,
        "parse_meta": parse_meta
    }

# =============================================================================
# Function 2 — Compute norms and select representative profile
# =============================================================================
def compute_norms_and_select(
    df_XT_centered: pd.DataFrame,
    df_XTd_centered: pd.DataFrame
) -> tuple[pd.DataFrame, Any]:
    """
    Compute Euclidean norms for each sample:
      - norm_T  = ||X_T_centered[i, :]||₂
      - norm_Td = ||X_Td_centered[i, :]||₂
      - norm_combined = ||[X_T_centered[i, :], X_Td_centered[i, :]]||₂

    Representative index = sample with minimal norm_combined.
    Returns:
      - df_norms: DataFrame with columns ['norm_T', 'norm_Td', 'norm_combined'], indexed by sample ID.
      - rep_idx: the selected representative sample index (preserves the CSV row index dtype).
    """
    XT = df_XT_centered.to_numpy(dtype=float)
    XTd = df_XTd_centered.to_numpy(dtype=float)

    norm_T = np.linalg.norm(XT, axis=1)
    norm_Td = np.linalg.norm(XTd, axis=1)
    combined = np.concatenate([XT, XTd], axis=1)
    norm_combined = np.linalg.norm(combined, axis=1)

    df_norms = pd.DataFrame({
        "norm_T": norm_T,
        "norm_Td": norm_Td,
        "norm_combined": norm_combined
    }, index=df_XT_centered.index)

    rep_pos = int(np.argmin(norm_combined))
    rep_idx = df_XT_centered.index[rep_pos]
    return df_norms, rep_idx

# =============================================================================
# Function 3 — Save outputs (profiles, anomalies, norms)
# =============================================================================
def write_outputs(
    output_dir: Path,
    rep_idx: Any,
    df_norms: pd.DataFrame,
    df_T_atm: pd.DataFrame,
    df_T_dp: pd.DataFrame,
    mu_T: np.ndarray,
    mu_Td: np.ndarray,
    H_2km: np.ndarray
) -> dict:
    """
    Save all Step 4 outputs to CSV files with headers and row indices.

    Files written:
      1) Norms_All_Samples.csv
         - Columns: norm_T, norm_Td, norm_combined
         - Row index: sample ID

      2) Representative_Sample_Index.csv
         - One-row CSV containing:
             rep_index, norm_T, norm_Td, norm_combined
         - Row index: [rep_index] for clarity

      3) Representative_T_profile_2km.csv
         - Columns: H, T_atm
         - Row index: 0..20

      4) Representative_Td_profile_2km.csv
         - Columns: H, T_DP
         - Row index: 0..20

      5) Representative_Anomaly_T_profile_2km.csv
         - Columns: H, T_anomaly (T_atm − μ_T)
         - Row index: 0..20

      6) Representative_Anomaly_Td_profile_2km.csv
         - Columns: H, Td_anomaly (T_DP − μ_Td)
         - Row index: 0..20

      7) Representative_Profile_Wide_2km.csv
         - Columns: H, T_atm, T_DP
         - Row index: 0..20
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Norms for all samples
    fn_norms = output_dir / "Norms_All_Samples.csv"
    df_norms.to_csv(fn_norms, index=True)

    # 2) Representative index (preserve its norms)
    rep_norms_row = df_norms.loc[[rep_idx]]
    fn_rep_idx = output_dir / "Representative_Sample_Index.csv"
    rep_norms_row.index.name = "rep_index"
    rep_norms_row.to_csv(fn_rep_idx, index=True)

    # Extract representative profiles as numpy arrays
    rep_T_values = df_T_atm.loc[rep_idx].to_numpy(dtype=float)
    rep_Td_values = df_T_dp.loc[rep_idx].to_numpy(dtype=float)

    # 3) Representative temperature profile (with H)
    df_rep_T = pd.DataFrame({
        "H": H_2km,
        "T_atm": rep_T_values
    })
    fn_rep_T = output_dir / "Representative_T_profile_2km.csv"
    df_rep_T.to_csv(fn_rep_T, index=True)

    # 4) Representative dew-point profile (with H)
    df_rep_Td = pd.DataFrame({
        "H": H_2km,
        "T_DP": rep_Td_values
    })
    fn_rep_Td = output_dir / "Representative_Td_profile_2km.csv"
    df_rep_Td.to_csv(fn_rep_Td, index=True)

    # 5) Temperature anomaly wrt μ_T
    df_rep_T_anom = pd.DataFrame({
        "H": H_2km,
        "T_anomaly": rep_T_values - mu_T
    })
    fn_rep_T_anom = output_dir / "Representative_Anomaly_T_profile_2km.csv"
    df_rep_T_anom.to_csv(fn_rep_T_anom, index=True)

    # 6) Dew-point anomaly wrt μ_Td
    df_rep_Td_anom = pd.DataFrame({
        "H": H_2km,
        "Td_anomaly": rep_Td_values - mu_Td
    })
    fn_rep_Td_anom = output_dir / "Representative_Anomaly_Td_profile_2km.csv"
    df_rep_Td_anom.to_csv(fn_rep_Td_anom, index=True)

    # 7) Wide view
    df_rep_wide = pd.DataFrame({
        "H": H_2km,
        "T_atm": rep_T_values,
        "T_DP": rep_Td_values
    })
    fn_rep_wide = output_dir / "Representative_Profile_Wide_2km.csv"
    df_rep_wide.to_csv(fn_rep_wide, index=True)

    return {
        "Norms_All_Samples": str(fn_norms),
        "Representative_Sample_Index": str(fn_rep_idx),
        "Representative_T_profile_2km": str(fn_rep_T),
        "Representative_Td_profile_2km": str(fn_rep_Td),
        "Representative_Anomaly_T_profile_2km": str(fn_rep_T_anom),
        "Representative_Anomaly_Td_profile_2km": str(fn_rep_Td_anom),
        "Representative_Profile_Wide_2km": str(fn_rep_wide),
    }

# =============================================================================
# Function 4 — Build a detailed processing log record
# =============================================================================
def build_log(
    log_path: Path,
    inputs: dict,
    outputs: dict,
    rep_idx: Any,
    df_norms: pd.DataFrame
) -> None:
    """
    Write a detailed log (text) describing what was done, how vectors were parsed, and where outputs are saved.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("Ground-based Atmospheric Profiles — Step 4: Representative Profile Selection")
    lines.append(f"Timestamp: {now}")
    lines.append("")
    lines.append("Inputs (from Step 1):")
    lines.append(f"  - X_T_centered: shape {inputs['df_XT_centered'].shape} (file: {INPUT_DIR_STEP1 / FN_X_T_CENTERED})")
    lines.append(f"  - X_Td_centered: shape {inputs['df_XTd_centered'].shape} (file: {INPUT_DIR_STEP1 / FN_X_TD_CENTERED})")
    lines.append(f"  - T_atm_2km: shape {inputs['df_T_atm'].shape} (file: {INPUT_DIR_STEP1 / FN_T_ATM_2KM})")
    lines.append(f"  - T_DP_2km: shape {inputs['df_T_dp'].shape} (file: {INPUT_DIR_STEP1 / FN_T_DP_2KM})")
    lines.append(f"  - μ_T (mean temperature; length 21): {INPUT_DIR_STEP1 / FN_MU_T}")
    lines.append(f"      Parsed as: {inputs['parse_meta']['mu_T'].get('source_layout', 'unknown')}")
    lines.append(f"      Attempts: {inputs['parse_meta']['mu_T'].get('attempts', [])}")
    lines.append(f"  - μ_Td (mean dew point; length 21): {INPUT_DIR_STEP1 / FN_MU_TD}")
    lines.append(f"      Parsed as: {inputs['parse_meta']['mu_Td'].get('source_layout', 'unknown')}")
    lines.append(f"      Attempts: {inputs['parse_meta']['mu_Td'].get('attempts', [])}")
    lines.append(f"  - H (first 21 used): {INPUT_DIR_STEP1 / FN_H}")
    lines.append(f"      Parsed as: {inputs['parse_meta']['H'].get('source_layout', 'unknown')}")
    lines.append(f"      Attempts: {inputs['parse_meta']['H'].get('attempts', [])}")
    lines.append("")
    lines.append("Method:")
    lines.append("  - For each sample i, compute Euclidean norms over 0–2 km:")
    lines.append("      norm_T(i)  = ||X_T_centered[i, :]||₂")
    lines.append("      norm_Td(i) = ||X_Td_centered[i, :]||₂")
    lines.append("      norm_combined(i) = ||[X_T_centered[i, :], X_Td_centered[i, :]]||₂")
    lines.append("  - Select representative sample as argmin_i norm_combined(i).")
    lines.append("  - This is equivalent to selecting the profile closest to the mean in PCA coefficient space,")
    lines.append("    since PCA uses an orthonormal basis (no need to actually perform PCA for Step 4).")
    lines.append("")
    lines.append(f"Representative sample index (row index from Step 1 CSVs): {rep_idx}")
    lines.append(f"  - norm_T: {df_norms.loc[rep_idx, 'norm_T']:.6e}")
    lines.append(f"  - norm_Td: {df_norms.loc[rep_idx, 'norm_Td']:.6e}")
    lines.append(f"  - norm_combined: {df_norms.loc[rep_idx, 'norm_combined']:.6e}")
    lines.append("")
    lines.append("Outputs (saved to: " + str(OUTPUT_DIR_STEP4) + ")")
    lines.append("  - Norms_All_Samples.csv")
    lines.append("      Columns: norm_T, norm_Td, norm_combined; Row index: sample ID.")
    lines.append("      Purpose: Provides per-sample distances to the mean profile in T, Td, and combined spaces.")
    lines.append("  - Representative_Sample_Index.csv")
    lines.append("      Single-row CSV keyed by rep_index; includes norm_T, norm_Td, norm_combined for the chosen sample.")
    lines.append("  - Representative_T_profile_2km.csv")
    lines.append("      Columns: H, T_atm; Row index: 0..20. The selected temperature profile over 0–2 km.")
    lines.append("  - Representative_Td_profile_2km.csv")
    lines.append("      Columns: H, T_DP; Row index: 0..20. The selected dew-point profile over 0–2 km.")
    lines.append("  - Representative_Anomaly_T_profile_2km.csv")
    lines.append("      Columns: H, T_anomaly = T_atm − μ_T; Row index: 0..20.")
    lines.append("  - Representative_Anomaly_Td_profile_2km.csv")
    lines.append("      Columns: H, Td_anomaly = T_DP − μ_Td; Row index: 0..20.")
    lines.append("  - Representative_Profile_Wide_2km.csv")
    lines.append("      Columns: H, T_atm, T_DP; Row index: 0..20. Convenience file combining T and Td.")
    lines.append("")
    lines.append("Notes:")
    lines.append("  - All CSVs include column headers and row indices.")
    lines.append("  - This step does not include any detector system effects (as requested).")
    lines.append("  - The selected representative profile is a real observed sample (not a synthetic mean).")

    log_path.write_text("\n".join(lines), encoding="utf-8")

# =============================================================================
# Function 5 — Main program (how to call Step 4)
# =============================================================================
def main():
    # Load inputs from Step 1 (robust CSV parsing)
    inputs = load_step1_data(INPUT_DIR_STEP1)

    # Compute norms and select representative
    df_norms, rep_idx = compute_norms_and_select(
        inputs["df_XT_centered"],
        inputs["df_XTd_centered"]
    )

    # Write outputs to Step 4 folder (without timestamp)
    outputs = write_outputs(
        OUTPUT_DIR_STEP4,
        rep_idx,
        df_norms,
        inputs["df_T_atm"],
        inputs["df_T_dp"],
        inputs["mu_T"],
        inputs["mu_Td"],
        inputs["H_2km"]
    )

    # Build detailed log record
    log_file = OUTPUT_DIR_STEP4 / "step4_processing_inventory.log"
    build_log(
        log_file,
        inputs,
        outputs,
        rep_idx,
        df_norms
    )

if __name__ == "__main__":
    main()