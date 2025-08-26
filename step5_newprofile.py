# (Main Script) Step 5 — Generate Perturbed Profiles from PCA Modes
# Author: (Your Name)
# Description:
#   Implements Step 5 of the project proposal: generate perturbed temperature (T) and dew-point (Td)
#   profiles by adding Δc_i · w_i to the representative profile for each retained principal component.
#   Inputs are read from Steps 1–4 outputs; outputs are written under "Step5_Perturbation" (no timestamp).
#
# How to use:
#   - Ensure the BASE_DIR points to the root where Step1_Processed, Step2_PCA, Step3_Stats,
#     Step4_RepresentativeProfile exist (as in your logs).
#   - Run this script directly, or import and call main().

from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd


# =============================================================================
# Global configuration (edit BASE_DIR if your directory structure is different)
# =============================================================================
BASE_DIR = "./"

DIR_STEP1 = os.path.join(BASE_DIR, "Step1_Processed")
DIR_STEP2 = os.path.join(BASE_DIR, "Step2_PCA")
DIR_STEP3 = os.path.join(BASE_DIR, "Step3_Stats")
DIR_STEP4 = os.path.join(BASE_DIR, "Step4_RepresentativeProfile")

# Output root for this step (no timestamp as requested)
DIR_STEP5 = os.path.join(BASE_DIR, "Step5_Perturbation")

# Physics-aware option: enforce dew-point <= temperature after dew point perturbations
ENFORCE_TD_LE_T = True

# Dataset configurations for this step
DATASETS = [
    # (label, variable_family, pca_subdir_name_in_Step2, stats_subdir_name_in_Step3)
    ("T_k95",  "T",  "PCA_T_W_k95.csv",  os.path.join("T_k95",  "delta_c_recommended.csv")),
    ("T_k99",  "T",  "PCA_T_W_k99.csv",  os.path.join("T_k99",  "delta_c_recommended.csv")),
    ("Td_k95", "Td", "PCA_Td_W_k95.csv", os.path.join("Td_k95", "delta_c_recommended.csv")),
    ("Td_k99", "Td", "PCA_Td_W_k99.csv", os.path.join("Td_k99", "delta_c_recommended.csv")),
]


# =============================================================================
# (F1) Utility — Robust loader for single-column vectors and standard CSVs
# =============================================================================
def load_vector_csv(path: str, expected_name: str | None = None) -> pd.Series:
    """
    Load a single-column CSV as a pandas Series. The file is expected to have a header.
    If 'expected_name' is provided, ensure the column is present (otherwise raise).
    Returns a Series with default integer index if original had no index column.

    Parameters:
      path          : Path to CSV.
      expected_name : Expected column name (e.g., 'mu_T', 'mu_Td', 'H').

    Returns:
      pd.Series with the single column.
    """
    df = pd.read_csv(path)
    if df.shape[1] != 1:
        # Try reading with index_col=0 (some writers include an index)
        df = pd.read_csv(path, index_col=0)
    if df.shape[1] != 1:
        raise ValueError(f"Expected single-column CSV at {path}, got shape {df.shape}")
    col = df.columns[0]
    if expected_name is not None and col != expected_name:
        # Allow flexibility but warn by renaming to expected for consistency
        df = df.rename(columns={col: expected_name})
        col = expected_name
    return df[col]


# =============================================================================
# (F2) Utility — Load W (PC loadings) and Δc_i (recommended coefficient differentials) with robust matching
# =============================================================================
def load_W_and_delta(
    variable_family: str,         # 'T' or 'Td'
    W_filename: str,              # e.g., 'PCA_T_W_k95.csv'
    delta_relpath: str            # e.g., 'T_k95/delta_c_recommended.csv'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load PC loadings W (21 × k) and the recommended Δc_i per PC (length k),
    and align Δc_i to W's columns (PC1..PCk).

    Returns:
      W_df (DataFrame): index -> height (21 rows), columns -> PC1..PCk
      delta_s (Series): index -> PC1..PCk, values -> Δc_i
    """
    # Load W
    W_path = os.path.join(DIR_STEP2, W_filename)
    W_df = pd.read_csv(W_path, index_col=0)  # rows indexed by heights, columns PC1..PCk
    # Validate shape
    if W_df.shape[0] != 21:
        # It should be 21 (0–2 km); if 21 is not found, try to slice first 21
        W_df = W_df.iloc[:21, :]
    # Load Δc
    delta_path = os.path.join(DIR_STEP3, delta_relpath)
    delta_df = pd.read_csv(delta_path, index_col=0)

    # Robust parsing to Series indexed by 'PC1'.. etc.
    if 'delta_c' in delta_df.columns:
        # Two common possibilities:
        #   (a) index=PC labels, single column delta_c
        #   (b) a normal table with a 'PC' column and 'delta_c' column
        if 'PC' in delta_df.columns:
            delta_s = delta_df.set_index('PC')['delta_c']
        else:
            delta_s = delta_df['delta_c']
    else:
        # Another pattern: a single row with columns 'PC1','PC2',...
        # or a single column with any name (e.g., 'bin_width') indexed by PC labels
        if all(col.startswith("PC") for col in delta_df.columns) and delta_df.shape[0] == 1:
            delta_s = delta_df.iloc[0]
        elif delta_df.shape[1] == 1:
            delta_s = delta_df.iloc[:, 0]
        else:
            raise ValueError(f"Unrecognized Δc format at {delta_path} with shape {delta_df.shape}")

    # Align to W columns; if indices not 'PCx', attempt to coerce
    def coerce_pc_labels(idx: pd.Index) -> List[str]:
        labels = []
        for k, item in enumerate(idx, start=1):
            s = str(item)
            if s.startswith("PC"):
                labels.append(s)
            else:
                labels.append(f"PC{k}")
        return labels

    # If delta index doesn't match W columns, coerce and reindex
    if not set(delta_s.index) == set(W_df.columns):
        # Try to coerce labels
        delta_s.index = coerce_pc_labels(delta_s.index)
    delta_s = delta_s.reindex(W_df.columns)
    if delta_s.isna().any():
        raise ValueError("Δc could not be aligned to W columns; please verify naming consistency.")

    return W_df, delta_s


# =============================================================================
# (F3) Loader — Representative profiles (T and Td) and H, μ_T, μ_Td
# =============================================================================
def load_representative_and_context() -> Dict[str, pd.Series]:
    """
    Load:
      - Representative temperature profile over 0–2 km (21) -> 'T_rep'
      - Representative dew-point profile over 0–2 km (21) -> 'Td_rep'
      - Height grid (first 21) -> 'H'
      - Mean temperature μ_T (21) -> 'mu_T'
      - Mean dew-point μ_Td (21) -> 'mu_Td'
    Returns a dict of Series.
    """
    # Representative profiles from Step 4
    rep_T_path = os.path.join(DIR_STEP4, "Representative_T_profile_2km.csv")
    rep_Td_path = os.path.join(DIR_STEP4, "Representative_Td_profile_2km.csv")
    rep_T_df = pd.read_csv(rep_T_path)
    rep_Td_df = pd.read_csv(rep_Td_path)
    # Ensure standard column names
    assert "H" in rep_T_df.columns and "T_atm" in rep_T_df.columns
    assert "H" in rep_Td_df.columns and "T_DP" in rep_Td_df.columns

    # Extract Series (aligned by row order 0..20)
    H = rep_T_df["H"]
    T_rep = rep_T_df["T_atm"]
    Td_rep = rep_Td_df["T_DP"]

    # μ vectors from Step 1
    mu_T = load_vector_csv(os.path.join(DIR_STEP1, "mu_T.csv"), expected_name="mu_T")
    mu_Td = load_vector_csv(os.path.join(DIR_STEP1, "mu_Td.csv"), expected_name="mu_Td")

    # H full from Step 1 (we keep the Step 4 H to match representative CSV)
    # But we can validate consistency if desired
    return {
        "H": H.reset_index(drop=True),
        "T_rep": T_rep.reset_index(drop=True),
        "Td_rep": Td_rep.reset_index(drop=True),
        "mu_T": mu_T.reset_index(drop=True),
        "mu_Td": mu_Td.reset_index(drop=True),
    }


# =============================================================================
# (F4) Engine — Generate per-PC perturbations for a single variable (T or Td)
# =============================================================================
def generate_perturbations_single_variable(
    rep_profile: pd.Series,  # T_rep or Td_rep
    W_df: pd.DataFrame,      # (21 × k)
    delta_s: pd.Series       # (k,)
) -> Dict[str, pd.DataFrame]:
    """
    For each PC i (column of W_df), compute:
      - plus:  rep + Δc_i * w_i
      - minus: rep - Δc_i * w_i
      - delta_plus:  +Δc_i * w_i
      - delta_minus: -Δc_i * w_i

    Returns a dict with:
      - 'plus'  : DataFrame (21 × k) columns PC1..PCk
      - 'minus' : DataFrame (21 × k)
      - 'delta_plus'  : DataFrame (21 × k)
      - 'delta_minus' : DataFrame (21 × k)
    """
    k = W_df.shape[1]
    pcs = list(W_df.columns)

    delta_matrix = W_df.values * delta_s.values  # broadcasting (21 × k)
    plus = rep_profile.values.reshape(-1, 1) + delta_matrix
    minus = rep_profile.values.reshape(-1, 1) - delta_matrix

    plus_df = pd.DataFrame(plus, columns=pcs)
    minus_df = pd.DataFrame(minus, columns=pcs)
    delta_plus_df = pd.DataFrame(delta_matrix, columns=pcs)
    delta_minus_df = pd.DataFrame(-delta_matrix, columns=pcs)

    return {
        "plus": plus_df,
        "minus": minus_df,
        "delta_plus": delta_plus_df,
        "delta_minus": delta_minus_df,
    }


# =============================================================================
# (F5) I/O — Write outputs for a single variable dataset (T_k95, T_k99, Td_k95, Td_k99)
# =============================================================================
def write_outputs_for_variable(
    out_dir: str,
    H: pd.Series,
    rep_profile: pd.Series,
    W_df: pd.DataFrame,
    delta_s: pd.Series,
    perturbations: Dict[str, pd.DataFrame],
    variable_family: str,   # 'T' or 'Td'
    paired_temperature_rep: pd.Series | None = None  # only used to enforce Td ≤ T
) -> Dict[str, str]:
    """
    Save:
      - Representative profile CSV
      - W modes CSV (copy of input)
      - delta_c_used.csv
      - Perturbed profiles (plus/minus) with H
      - Delta profiles (plus/minus) with H
      - Per-PC detailed CSVs under per_PC/

    Returns:
      A dict mapping file roles to file paths (for logging).
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "per_PC"), exist_ok=True)

    # Prepare DataFrames with H
    rep_df = pd.DataFrame({"H": H, f"{variable_family}_rep": rep_profile})
    W_out = W_df.copy()
    W_out.index = H  # index by height for readability

    delta_used_df = pd.DataFrame({"PC": delta_s.index, "delta_c": delta_s.values})

    # Optionally enforce Td ≤ T for dew-point perturbations
    def enforce_td_le_t(df: pd.DataFrame, T_ref: pd.Series) -> pd.DataFrame:
        """Clip dew-point columns so that Td ≤ T_ref at each level."""
        return df.apply(lambda col: np.minimum(col.values, T_ref.values), axis=0, result_type='expand')

    plus_df = perturbations["plus"].copy()
    minus_df = perturbations["minus"].copy()
    delta_plus_df = perturbations["delta_plus"].copy()
    delta_minus_df = perturbations["delta_minus"].copy()

    if variable_family == "Td" and ENFORCE_TD_LE_T and paired_temperature_rep is not None:
        plus_df = enforce_td_le_t(plus_df, paired_temperature_rep)
        minus_df = enforce_td_le_t(minus_df, paired_temperature_rep)
        # deltas become state-dependent if clipping occurs; we still save the unclipped theoretical deltas separately
        # but also provide "delta_effective" as (perturbed - rep)
        delta_plus_eff = plus_df.values - rep_profile.values.reshape(-1, 1)
        delta_minus_eff = minus_df.values - rep_profile.values.reshape(-1, 1)
        delta_plus_eff_df = pd.DataFrame(delta_plus_eff, columns=plus_df.columns)
        delta_minus_eff_df = pd.DataFrame(delta_minus_eff, columns=minus_df.columns)
    else:
        delta_plus_eff_df = delta_plus_df.copy()
        delta_minus_eff_df = delta_minus_df.copy()

    # Re-index all by height for readability
    for df in (plus_df, minus_df, delta_plus_df, delta_minus_df, delta_plus_eff_df, delta_minus_eff_df):
        df.index = H

    # Write files
    files_out = {}
    p = os.path.join

    # Representative
    rep_path = p(out_dir, f"Representative_{variable_family}_profile_2km.csv")
    rep_df.to_csv(rep_path, index=True)
    files_out["representative_profile"] = rep_path

    # W modes
    W_path = p(out_dir, "W_modes.csv")
    W_out.to_csv(W_path, index=True)
    files_out["W_modes"] = W_path

    # delta_c used
    delta_used_path = p(out_dir, "delta_c_used.csv")
    delta_used_df.to_csv(delta_used_path, index=False)
    files_out["delta_c_used"] = delta_used_path

    # Bulk matrices
    plus_path = p(out_dir, "Perturbed_profiles_plus.csv")
    minus_path = p(out_dir, "Perturbed_profiles_minus.csv")
    dplus_path = p(out_dir, "Delta_profiles_plus_theoretical.csv")
    dminus_path = p(out_dir, "Delta_profiles_minus_theoretical.csv")
    dplus_eff_path = p(out_dir, "Delta_profiles_plus_effective.csv")
    dminus_eff_path = p(out_dir, "Delta_profiles_minus_effective.csv")

    plus_df.to_csv(plus_path, index=True)
    minus_df.to_csv(minus_path, index=True)
    delta_plus_df.to_csv(dplus_path, index=True)
    delta_minus_df.to_csv(dminus_path, index=True)
    delta_plus_eff_df.to_csv(dplus_eff_path, index=True)
    delta_minus_eff_df.to_csv(dminus_eff_path, index=True)

    files_out["profiles_plus"] = plus_path
    files_out["profiles_minus"] = minus_path
    files_out["delta_plus_theoretical"] = dplus_path
    files_out["delta_minus_theoretical"] = dminus_path
    files_out["delta_plus_effective"] = dplus_eff_path
    files_out["delta_minus_effective"] = dminus_eff_path

    # Per-PC files
    for pc in W_df.columns:
        per_pc_df = pd.DataFrame({
            "H": H.values,
            f"{variable_family}_rep": rep_profile.values,
            f"{variable_family}_plus": plus_df[pc].values,
            f"{variable_family}_minus": minus_df[pc].values,
            "delta_c": np.full_like(H.values, delta_s[pc], dtype=float),
            "w_i": W_df[pc].values,
            "delta_theoretical_plus": delta_plus_df[pc].values,
            "delta_theoretical_minus": delta_minus_df[pc].values,
            "delta_effective_plus": delta_plus_eff_df[pc].values,
            "delta_effective_minus": delta_minus_eff_df[pc].values,
        })
        per_pc_path = p(out_dir, "per_PC", f"{pc}_perturbation.csv")
        per_pc_df.to_csv(per_pc_path, index=False)

    return files_out


# =============================================================================
# (F6) Logging — Build and write a human-readable inventory log of this step
# =============================================================================
def write_inventory_log(
    overall_outputs: Dict[str, Dict[str, str]],
    H: pd.Series,
    rep_T: pd.Series,
    rep_Td: pd.Series
) -> str:
    """
    Write a detailed log summarizing what was produced and where it was saved.

    overall_outputs: dict keyed by dataset label (e.g., 'T_k95'), values are dicts of file roles -> paths.
    Returns path to the log file.
    """
    os.makedirs(DIR_STEP5, exist_ok=True)
    log_path = os.path.join(DIR_STEP5, "Step5_Perturbation_inventory.log")

    lines = []
    lines.append("Ground-based Atmospheric Profiles — Step 5: Generate Perturbed Profiles")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Inputs used (from prior steps):")
    lines.append(f"  - Step 1: μ_T, μ_Td, H (first 21 levels) from: {DIR_STEP1}")
    lines.append(f"  - Step 2: W (PC loadings) from: {DIR_STEP2}")
    lines.append(f"  - Step 3: Δc_i (recommended bin-width coefficients) from: {DIR_STEP3}")
    lines.append(f"  - Step 4: Representative profiles from: {DIR_STEP4}")
    lines.append("")
    lines.append("Core relation and interpretation:")
    lines.append("  - X_perturbed = X_rep + Δc_i · w_i (for each PC i)")
    lines.append("    where w_i is the i-th PC loading (column of W), Δc_i is from Step 3.")
    lines.append("  - Temperature (T) and dew-point (Td) are perturbed separately.")
    lines.append(f"  - Dew-point clipping enabled: ENFORCE_TD_LE_T = {ENFORCE_TD_LE_T} (Td ≤ T applied level-wise).")
    lines.append("")
    lines.append(f"Representative profiles (length {len(H)} over 0–2 km):")
    lines.append("  - Representative_T_profile_2km.csv (columns: H, T_atm) [from Step 4]")
    lines.append("  - Representative_Td_profile_2km.csv (columns: H, T_DP) [from Step 4]")
    lines.append("")
    lines.append("Outputs generated in this step (by dataset):")
    for dataset_label, files in overall_outputs.items():
        lines.append(f"  Dataset: {dataset_label}")
        # Explanations for each file role
        role_explain = {
            "representative_profile": "Representative profile used as baseline (columns: H, var_rep).",
            "W_modes": "PC loadings W re-indexed by height (rows=H, columns=PC1..PCk).",
            "delta_c_used": "Δc_i values per PC actually used in perturbations.",
            "profiles_plus": "Perturbed profiles with +Δc_i along each PC (rows=H, cols=PC1..PCk).",
            "profiles_minus": "Perturbed profiles with −Δc_i along each PC (rows=H, cols=PC1..PCk).",
            "delta_plus_theoretical": "Theoretical deltas (+Δc_i·w_i) added to the representative (rows=H, cols=PCs).",
            "delta_minus_theoretical": "Theoretical deltas (−Δc_i·w_i) subtracted from the representative (rows=H, cols=PCs).",
            "delta_plus_effective": "Effective deltas after any physical clipping (Td only) (rows=H, cols=PCs).",
            "delta_minus_effective": "Effective deltas after any physical clipping (Td only) (rows=H, cols=PCs).",
        }
        for role, path in files.items():
            explain = role_explain.get(role, "Output file.")
            lines.append(f"    - {os.path.relpath(path, BASE_DIR)}")
            lines.append(f"        Meaning: {explain}")
        # Per-PC folder
        per_pc_dir = os.path.join(os.path.dirname(list(files.values())[0]), "per_PC")
        lines.append(f"    - {os.path.relpath(per_pc_dir, BASE_DIR)}\\PCi_perturbation.csv (one file per PC)")
        lines.append("        Meaning: Detailed per-PC CSV with H, rep, plus, minus, Δc, w_i, and delta (theoretical/effective).")
        lines.append("")
    lines.append("Notes:")
    lines.append("  - All CSVs include column headers; matrices indexed by height (H).")
    lines.append("  - Theoretical vs effective deltas differ only if dew-point clipping is applied.")
    lines.append("  - No detector system effects, thermal emission, or noise are included here (by design for Step 5).")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return log_path


# =============================================================================
# (F7) Main Orchestrator — Execute Step 5
# =============================================================================
def main():
    # Prepare output root
    os.makedirs(DIR_STEP5, exist_ok=True)

    # Load representative and context
    ctx = load_representative_and_context()
    H = ctx["H"]
    T_rep = ctx["T_rep"]
    Td_rep = ctx["Td_rep"]
    # mu_T, mu_Td are loaded for completeness; not strictly needed for perturbation in this implementation
    mu_T = ctx["mu_T"]
    mu_Td = ctx["mu_Td"]

    # Accumulate outputs for logging
    overall_outputs: Dict[str, Dict[str, str]] = {}

    # Loop over datasets
    for dataset_label, variable_family, W_filename, delta_relpath in DATASETS:
        # Load W and Δc
        W_df, delta_s = load_W_and_delta(variable_family, W_filename, delta_relpath)
        # Representative profile for this variable
        if variable_family == "T":
            rep_profile = T_rep
            paired_T = None  # not used for T
        else:
            rep_profile = Td_rep
            paired_T = T_rep  # for optional clipping Td ≤ T

        # Generate perturbations
        perturbations = generate_perturbations_single_variable(rep_profile, W_df, delta_s)

        # Output directory per dataset
        out_dir = os.path.join(DIR_STEP5, dataset_label)
        files_out = write_outputs_for_variable(
            out_dir=out_dir,
            H=H,
            rep_profile=rep_profile,
            W_df=W_df,
            delta_s=delta_s,
            perturbations=perturbations,
            variable_family=variable_family,
            paired_temperature_rep=paired_T
        )
        overall_outputs[dataset_label] = files_out

    # Write log
    log_path = write_inventory_log(overall_outputs, H=H, rep_T=T_rep, rep_Td=Td_rep)
    print(f"Step 5 completed. Inventory log written to: {log_path}")


if __name__ == "__main__":
    main()