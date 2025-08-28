# (Module: SJ3) Step 3 — Statistical Analysis for Joint PCA Coefficients (Seaborn-free)
# Title: step3_stats_joint.py
# Author: (Your Lab)
# Date: 2025-08-28
# Description:
#   Implements Step 3 (README.md) for the joint PCA case (concatenated T and Td).
#   Inputs (from Step 2 joint PCA):
#     - PCA_joint_scores_k95.csv / PCA_joint_scores_k99.csv (scores)
#     - PCA_joint_explained_variance_ratio.csv (full EVR)
#   Outputs (no timestamp in folder name):
#     - Step3_Stats_Joint/[Joint_k95|Joint_k99]/
#         per_PC_stats.csv
#         per_PC_binning_plan.csv
#         delta_c_recommended.csv
#         explained_variance_ratio_k.csv
#         kde_bandwidth_report.csv
#         histograms/PCi_histogram.csv
#         plots/PCi_pdf.png
#         plots/all_PCs_pdf_grid.png
#         plots_compare/PCi_bin_compare.png (FD/Scott/Sturges and scaled-FD overlays vs KDE)
#     - Step3_Stats_Joint/Step3_Stats_Joint_inventory.log
#
# Notes:
#   - No command-line parsing; edit USER_PATHS below and run run_step3_joint_main().
#   - This step analyzes only the coefficient distributions (C), independent of detector effects.
#   - KDE uses SciPy's gaussian_kde (Gaussian kernel; default Scott's rule for bandwidth).
#
# Core notation (readable fonts and meanings):
#   - C               : (n_samples × k) retained PCA scores (coefficients).
#   - EVR_k           : Explained variance ratio for the retained k PCs.
#   - Δc_i (delta_c)  : Recommended coefficient differential for PC i (selected histogram bin width).
#   - KDE kernel      : Gaussian; bandwidth h = factor × s with factor = n^(−1/5) for 1D Scott’s rule.
#   - FWHM (Gaussian) : FWHM = 2√(2 ln 2) × h ≈ 2.35482 × h, where h is the kernel standard deviation.
#
# How to call (main program):
#   1) Set STEP2_JOINT_DIR to your Step 2 joint output folder (the timestamped directory printed in Step2_JointPCA inventory).
#   2) Optionally adjust STEP3_JOINT_DIR (default: ./Step3_Stats_Joint; no timestamp).
#   3) In Python/VSCode, run: run_step3_joint_main()
#      (Do not use command-line parsing.)

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # for non-interactive environments
import matplotlib.pyplot as plt

# =========================
# USER_PATHS (edit here)
# =========================
# Point this to the Step 2 JOINT PCA output directory (timestamped folder you already have).
# Example from your log:
#   C:\Users\...\prior-analysis\Step2_JointPCA_20250828_224738
STEP2_JOINT_DIR = Path(r"C:\Users\24573\OneDrive - The University of Hong Kong\PhD\Project\atmospheric sounding\prior-analysis\Step2_JointPCA")

# Step 3 outputs will be saved here (no timestamp):
STEP3_JOINT_DIR = Path("./Step3_Stats_Joint")

# Datasets: joint k95 and k99
DATASETS = {
    "Joint_k95": ("PCA_joint_scores_k95.csv", "PCA_joint_explained_variance_ratio.csv"),
    "Joint_k99": ("PCA_joint_scores_k99.csv", "PCA_joint_explained_variance_ratio.csv"),
}

# Plot styling
plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8


# (SJ3.1) Function — Title: Safe creation of output directories
def ensure_output_dirs_joint(root: Path, dataset_keys: List[str]) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    subdirs = {}
    for k in dataset_keys:
        d = root / k
        (d / "histograms").mkdir(parents=True, exist_ok=True)
        (d / "plots").mkdir(parents=True, exist_ok=True)
        (d / "plots_compare").mkdir(parents=True, exist_ok=True)
        subdirs[k] = d
    return subdirs


# (SJ3.2) Function — Title: Load joint PCA scores and explained-variance ratios
def load_joint_scores_and_evr(step2_joint_dir: Path,
                              datasets: Dict[str, Tuple[str, str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    result = {}
    for key, (scores_csv, evr_csv) in datasets.items():
        scores_path = step2_joint_dir / scores_csv
        evr_path = step2_joint_dir / evr_csv
        if not scores_path.is_file():
            raise FileNotFoundError(f"Scores file not found for {key}: {scores_path}")
        if not evr_path.is_file():
            raise FileNotFoundError(f"Explained variance ratio file not found for {key}: {evr_path}")

        scores = pd.read_csv(scores_path, index_col=0)
        evr_df = pd.read_csv(evr_path, index_col=0, header=0)

        if evr_df.shape[1] == 1:
            evr_series = evr_df.iloc[:, 0]
        else:
            evr_series = evr_df["explained_variance_ratio"] if "explained_variance_ratio" in evr_df.columns else evr_df.iloc[:, 0]

        result[key] = {"scores": scores, "evr_full": evr_series}
    return result


# (SJ3.3) Function — Title: Compute descriptive statistics per PC
def compute_per_pc_stats(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in scores.columns:
        x = scores[col].to_numpy()
        n = x.size
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1)) if n > 1 else np.nan
        try:
            skew = float(stats.skew(x, bias=False)) if n > 2 else np.nan
        except Exception:
            skew, std = np.nan, std
        try:
            kurt = float(stats.kurtosis(x, fisher=True, bias=False)) if n > 3 else np.nan
        except Exception:
            kurt = np.nan

        pcts = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
        xmin, xmax = float(np.min(x)), float(np.max(x))

        try:
            k2_stat, k2_p = stats.normaltest(x)
        except Exception:
            k2_stat, k2_p = np.nan, np.nan

        rows.append({
            "PC": col,
            "mean": mean,
            "std": std,
            "skew": skew,
            "kurtosis": kurt,
            "min": xmin,
            "p01": pcts[0],
            "p05": pcts[1],
            "p25": pcts[2],
            "median": pcts[3],
            "p75": pcts[4],
            "p95": pcts[5],
            "p99": pcts[6],
            "max": xmax,
            "n": int(n),
            "normaltest_stat": float(k2_stat) if k2_stat is not np.nan else np.nan,
            "normaltest_p": float(k2_p) if k2_p is not np.nan else np.nan,
        })
    return pd.DataFrame(rows).set_index("PC")


# (SJ3.4) Function — Title: Compute histogram binning plan (FD, Scott, Sturges)
def compute_binning_plan(scores: pd.DataFrame,
                         min_bins: int = 5,
                         max_bins: int = 200) -> pd.DataFrame:
    plan_rows = []
    for col in scores.columns:
        x = scores[col].to_numpy()
        n = x.size
        xmin, xmax = float(np.min(x)), float(np.max(x))
        data_range = float(xmax - xmin) if xmax > xmin else 1.0
        q25, q75 = np.percentile(x, [25, 75])
        iqr = float(q75 - q25)
        std = float(np.std(x, ddof=1)) if n > 1 else 0.0

        # Freedman–Diaconis
        fd_width = 2.0 * iqr * (n ** (-1.0/3.0)) if iqr > 0 else np.nan
        fd_bins = int(np.ceil(data_range / fd_width)) if (np.isfinite(fd_width) and fd_width > 0) else np.nan

        # Scott
        scott_width = 3.5 * std * (n ** (-1.0/3.0)) if std > 0 else np.nan
        scott_bins = int(np.ceil(data_range / scott_width)) if (np.isfinite(scott_width) and scott_width > 0) else np.nan

        # Sturges
        sturges_bins = int(np.ceil(np.log2(n) + 1)) if n > 0 else 1

        # Recommendation with clamping
        if not np.isnan(fd_bins) and fd_bins > 0:
            recommended_rule = "FD"
            recommended_bins = int(max(min_bins, min(fd_bins, max_bins)))
            recommended_width = data_range / recommended_bins
        elif not np.isnan(scott_bins) and scott_bins > 0:
            recommended_rule = "Scott"
            recommended_bins = int(max(min_bins, min(scott_bins, max_bins)))
            recommended_width = data_range / recommended_bins
        else:
            recommended_rule = "Sturges"
            recommended_bins = int(max(min_bins, min(sturges_bins, max_bins)))
            recommended_width = data_range / recommended_bins

        plan_rows.append({
            "PC": col,
            "n": int(n),
            "data_min": xmin,
            "data_max": xmax,
            "IQR": float(iqr),
            "std": float(std),
            "FD_width": float(fd_width) if np.isfinite(fd_width) else np.nan,
            "FD_bins": int(fd_bins) if not np.isnan(fd_bins) else np.nan,
            "Scott_width": float(scott_width) if np.isfinite(scott_width) else np.nan,
            "Scott_bins": int(scott_bins) if not np.isnan(scott_bins) else np.nan,
            "Sturges_bins": int(sturges_bins),
            "recommended_rule": recommended_rule,
            "recommended_bins": int(recommended_bins),
            "recommended_width": float(recommended_width),
        })
    return pd.DataFrame(plan_rows).set_index("PC")


# (SJ3.5) Function — Title: Export histograms (CSV) and per-PC plots (PNG, KDE via SciPy)
def export_histograms_and_plots_joint(scores: pd.DataFrame,
                                      bin_plan: pd.DataFrame,
                                      out_hist_dir: Path,
                                      out_plots_dir: Path) -> List[Path]:
    created_files: List[Path] = []
    for col in scores.columns:
        x = scores[col].to_numpy()
        rec_bins = int(bin_plan.loc[col, "recommended_bins"])
        xmin = float(bin_plan.loc[col, "data_min"])
        xmax = float(bin_plan.loc[col, "data_max"])
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5

        counts, edges = np.histogram(x, bins=rec_bins, range=(xmin, xmax), density=False)
        densities, _ = np.histogram(x, bins=rec_bins, range=(xmin, xmax), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Save histogram CSV
        hist_df = pd.DataFrame({
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": centers,
            "count": counts,
            "density": densities
        })
        hist_csv_path = out_hist_dir / f"{col}_histogram.csv"
        hist_df.to_csv(hist_csv_path, index=False)
        created_files.append(hist_csv_path)

        # Plot
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.hist(x, bins=edges, density=True, alpha=0.35, color="tab:blue",
                edgecolor="black", linewidth=0.5, label="Histogram (density)")

        sigma = np.std(x, ddof=1) if x.size > 1 else 0.0
        if sigma > 0:
            kde = stats.gaussian_kde(x)  # Gaussian kernel, Scott's rule by default
            grid = np.linspace(edges[0], edges[-1], 300)
            kde_y = kde(grid)
            ax.plot(grid, kde_y, color="tab:orange", linewidth=1.5, label="KDE (Gaussian kernel)")

            mu = float(np.mean(x))
            pdf = stats.norm.pdf(grid, loc=mu, scale=sigma)
            ax.plot(grid, pdf, color="tab:green", linestyle="--", linewidth=1.3,
                    label=f"Normal fit μ={mu:.3g}, σ={sigma:.3g}")

        ax.set_title(f"{col} — Coefficient PDF (density)")
        ax.set_xlabel("Coefficient value")
        ax.set_ylabel("Density")
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        plot_path = out_plots_dir / f"{col}_pdf.png"
        fig.savefig(plot_path)
        plt.close(fig)
        created_files.append(plot_path)

    return created_files


# (SJ3.6) Function — Title: Export grid figures for all PCs in a dataset
def export_grid_plots_joint(scores: pd.DataFrame,
                            bin_plan: pd.DataFrame,
                            out_plots_dir: Path,
                            cols: int = 3) -> Path:
    pcs = list(scores.columns)
    n = len(pcs)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.6), squeeze=False)

    for i, col in enumerate(pcs):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        x = scores[col].to_numpy()

        rec_bins = int(bin_plan.loc[col, "recommended_bins"])
        xmin = float(bin_plan.loc[col, "data_min"])
        xmax = float(bin_plan.loc[col, "data_max"])
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5

        ax.hist(x, bins=rec_bins, range=(xmin, xmax), density=True,
                alpha=0.35, color="tab:blue", edgecolor="black", linewidth=0.4)

        sigma = np.std(x, ddof=1) if x.size > 1 else 0.0
        if sigma > 0:
            kde = stats.gaussian_kde(x)
            grid = np.linspace(xmin, xmax, 300)
            ax.plot(grid, kde(grid), color="tab:orange", linewidth=1.2)

        ax.set_title(col, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle("Joint PCA — Coefficient PDFs (all PCs)", y=1.02, fontsize=13)
    fig.tight_layout()
    out_path = out_plots_dir / "all_PCs_pdf_grid.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# (SJ3.7) Function — Title: Save summary CSVs (stats, bin plan, Δc_i, EVR_k)
def export_summaries_joint(stats_df: pd.DataFrame,
                           plan_df: pd.DataFrame,
                           evr_full: pd.Series,
                           scores_df: pd.DataFrame,
                           out_dir: Path) -> Dict[str, Path]:
    files = {}

    stats_csv = out_dir / "per_PC_stats.csv"
    stats_df.to_csv(stats_csv, index=True)
    files["per_PC_stats.csv"] = stats_csv

    plan_csv = out_dir / "per_PC_binning_plan.csv"
    plan_df.to_csv(plan_csv, index=True)
    files["per_PC_binning_plan.csv"] = plan_csv

    delta_c = plan_df[["recommended_rule", "recommended_bins", "recommended_width"]].copy()
    delta_c.rename(columns={"recommended_width": "delta_c"}, inplace=True)
    delta_c_csv = out_dir / "delta_c_recommended.csv"
    delta_c.to_csv(delta_c_csv, index=True)
    files["delta_c_recommended.csv"] = delta_c_csv

    k = scores_df.shape[1]
    evr_k = evr_full.iloc[:k].copy()
    evr_k.index = [f"PC{i+1}" for i in range(k)]
    evr_k.name = "explained_variance_ratio"
    evr_k_csv = out_dir / "explained_variance_ratio_k.csv"
    evr_k.to_csv(evr_k_csv, header=True, index=True)
    files["explained_variance_ratio_k.csv"] = evr_k_csv

    return files


# (SJ3.8) Function — Title: Compute KDE bandwidth report (kernel, bandwidth, FWHM)
def compute_kde_bandwidth_report(scores: pd.DataFrame) -> pd.DataFrame:
    """
    For 1D KDE with Gaussian kernel:
      - Scott's factor (default in SciPy): factor = n^(-1/5)
      - Bandwidth h = factor × s (s = sample std, ddof=1)
      - FWHM = 2.35482 × h
    """
    rows = []
    for col in scores.columns:
        x = scores[col].to_numpy()
        n = x.size
        s = float(np.std(x, ddof=1)) if n > 1 else np.nan
        if np.isfinite(s) and n > 1:
            scott_factor = n ** (-1.0 / 5.0)
            h_scott = scott_factor * s
            fwhm_scott = 2.35482004503 * h_scott
            # Also report Silverman ROT for reference
            sigma_a = s
            iqr = np.subtract(*np.percentile(x, [75, 25]))
            sigma_b = iqr / 1.34 if iqr > 0 else sigma_a
            sigma_star = min(sigma_a, sigma_b)
            h_silverman = 0.9 * sigma_star * n ** (-1.0 / 5.0)
            fwhm_silverman = 2.35482004503 * h_silverman
        else:
            scott_factor = np.nan
            h_scott = np.nan
            fwhm_scott = np.nan
            h_silverman = np.nan
            fwhm_silverman = np.nan

        rows.append({
            "PC": col,
            "n": int(n),
            "std": s,
            "kernel": "Gaussian",
            "bandwidth_rule": "Scott (default in SciPy gaussian_kde)",
            "bandwidth_h_scott": h_scott,
            "FWHM_scott": fwhm_scott,
            "bandwidth_h_silverman": h_silverman,
            "FWHM_silverman": fwhm_silverman,
            "scott_factor": scott_factor
        })
    return pd.DataFrame(rows).set_index("PC")


# (SJ3.9) Function — Title: Export bin-size comparison plots vs KDE
def export_bin_size_comparison_plots(scores: pd.DataFrame,
                                     out_plots_dir: Path,
                                     compare_width_multipliers: List[float] = [1.0, 0.75, 0.5, 0.25],
                                     include_rules: List[str] = ["FD", "Scott", "Sturges"]) -> List[Path]:
    """
    For each PC, create a panel figure comparing histograms for:
      - FD, Scott, Sturges base rules, and
      - Scaled-FD widths (e.g., 1.0×, 0.75×, 0.5×, 0.25× of the FD bin width),
    all overlaid with the KDE curve (Gaussian kernel, Scott bandwidth).
    """
    created: List[Path] = []

    for col in scores.columns:
        x = scores[col].to_numpy()
        n = x.size
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        data_range = float(xmax - xmin)

        # Base widths (may be NaN if degenerate)
        q25, q75 = np.percentile(x, [25, 75])
        iqr = float(q75 - q25)
        s = float(np.std(x, ddof=1)) if n > 1 else 0.0

        fd_width = 2.0 * iqr * (n ** (-1.0/3.0)) if iqr > 0 else np.nan
        scott_width = 3.5 * s * (n ** (-1.0/3.0)) if s > 0 else np.nan
        sturges_bins = int(np.ceil(np.log2(n) + 1)) if n > 0 else 1
        sturges_width = data_range / max(1, sturges_bins)

        # Prepare figure
        fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), constrained_layout=True, sharey=True)
        axes = axes.ravel()

        # Panel 1: FD (if available)
        if np.isfinite(fd_width) and fd_width > 0 and "FD" in include_rules:
            bins = int(max(5, np.ceil(data_range / fd_width)))
            edges = np.linspace(xmin, xmax, bins + 1)
            axes[0].hist(x, bins=edges, density=True, alpha=0.35, color="tab:blue", edgecolor="black", linewidth=0.5)
            axes[0].set_title(f"FD: bins={bins}, width≈{data_range/bins:.3g}")
        else:
            axes[0].axis("off")

        # Panel 2: Scott (if available)
        if np.isfinite(scott_width) and scott_width > 0 and "Scott" in include_rules:
            bins = int(max(5, np.ceil(data_range / scott_width)))
            edges = np.linspace(xmin, xmax, bins + 1)
            axes[1].hist(x, bins=edges, density=True, alpha=0.35, color="tab:blue", edgecolor="black", linewidth=0.5)
            axes[1].set_title(f"Scott: bins={bins}, width≈{data_range/bins:.3g}")
        else:
            axes[1].axis("off")

        # Panel 3: Sturges
        if "Sturges" in include_rules:
            bins = max(5, sturges_bins)
            edges = np.linspace(xmin, xmax, bins + 1)
            axes[2].hist(x, bins=edges, density=True, alpha=0.35, color="tab:blue", edgecolor="black", linewidth=0.5)
            axes[2].set_title(f"Sturges: bins={bins}, width≈{data_range/bins:.3g}")

        # Panels 4-8: scaled FD widths
        grid_x = np.linspace(xmin, xmax, 300)
        kde = stats.gaussian_kde(x) if s > 0 else None
        for j, mult in enumerate(compare_width_multipliers[:4], start=3):
            if np.isfinite(fd_width) and fd_width > 0:
                width_j = fd_width * mult
            else:
                # fallback to Scott if FD not available
                width_j = scott_width * mult if np.isfinite(scott_width) and scott_width > 0 else sturges_width * mult

            bins = int(max(5, np.ceil(data_range / width_j))) if width_j > 0 else 5
            edges = np.linspace(xmin, xmax, bins + 1)
            axes[j+1].hist(x, bins=edges, density=True, alpha=0.35, color="tab:blue",
                         edgecolor="black", linewidth=0.5)
            axes[j+1].set_title(f"Scaled width ×{mult:g}: bins={bins}, width≈{data_range/bins:.3g}")

        # Overlay KDE on all active panels
        if kde is not None:
            kde_y = kde(grid_x)
            for ax in axes:
                if ax.has_data():
                    ax.plot(grid_x, kde_y, color="tab:orange", linewidth=1.3, label="KDE")
                    # Show normal fit too
                    mu = float(np.mean(x))
                    sigma = float(np.std(x, ddof=1))
                    ax.plot(grid_x, stats.norm.pdf(grid_x, mu, sigma),
                            color="tab:green", linestyle="--", linewidth=1.1, label="Normal fit")
                    ax.legend(loc="best", frameon=True)

        for ax in axes:
            if ax.has_data():
                ax.set_xlabel("Coefficient value")
                ax.set_ylabel("Density")

        fig.suptitle(f"{col} — Histogram bin-size comparison vs KDE", fontsize=13)
        out_path = out_plots_dir / "plots_compare" / f"{col}_bin_compare.png"
        fig.savefig(out_path)
        plt.close(fig)
        created.append(out_path)

    return created


# (SJ3.10) Function — Title: Write detailed inventory log (what and where)
def write_inventory_log_joint(root_dir: Path,
                              created_map: Dict[str, Dict[str, List[Path]]],
                              summary_map: Dict[str, Dict[str, Path]],
                              context: Dict[str, str]) -> Path:
    log_path = root_dir / "Step3_Stats_Joint_inventory.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("Ground-based Atmospheric Profiles — Step 3 (Joint): Statistical Analysis of PCA Coefficients")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("")
    lines.append("Inputs (from Step 2 Joint PCA):")
    lines.append(f"  Step 2 Joint directory: {context.get('step2_joint_dir','')}")
    lines.append("  - PCA_joint_scores_k95.csv / PCA_joint_scores_k99.csv: Truncated joint scores C = X_joint_centered · W_joint.")
    lines.append("  - PCA_joint_explained_variance_ratio.csv: Full explained variance ratios per PC (sklearn definition).")
    lines.append("")
    lines.append("Core quantities:")
    lines.append("  - C (scores): Principal component coefficients matrix for the retained PCs.")
    lines.append("  - Δc_i (delta_c): Recommended coefficient differential per PC (selected histogram bin width).")
    lines.append("  - KDE: Gaussian kernel with Scott bandwidth; FWHM = 2.35482 × h, h = n^(−1/5) × std.")
    lines.append("")
    lines.append("Outputs (this step):")
    for key in created_map:
        lines.append(f"  Dataset: {key}")
        sm = summary_map.get(key, {})
        if sm:
            lines.append("    Summary CSVs:")
            if "per_PC_stats.csv" in sm:
                lines.append(f"      - {sm['per_PC_stats.csv']}")
                lines.append("        Meaning: Per-PC descriptive statistics (mean, std, skewness, kurtosis, percentiles, normality test).")
            if "per_PC_binning_plan.csv" in sm:
                lines.append(f"      - {sm['per_PC_binning_plan.csv']}")
                lines.append("        Meaning: FD/Scott/Sturges binning plan; recommended rule, bins, and bin width.")
            if "delta_c_recommended.csv" in sm:
                lines.append(f"      - {sm['delta_c_recommended.csv']}")
                lines.append("        Meaning: Recommended Δc_i per PC (equal to recommended histogram bin width).")
            if "explained_variance_ratio_k.csv" in sm:
                lines.append(f"      - {sm['explained_variance_ratio_k.csv']}")
                lines.append("        Meaning: Explained variance ratios for the retained PCs (truncated to k).")
            kde_rep = sm.get("kde_bandwidth_report.csv")
            if kde_rep:
                lines.append(f"      - {kde_rep}")
                lines.append("        Meaning: KDE kernel, Scott/Silverman bandwidths, and Gaussian FWHM per PC.")
        cr = created_map[key]
        if cr.get("histograms_and_plots"):
            lines.append("    Histograms and per-PC plots:")
            lines.append(f"      - CSV histograms (bin_left, bin_right, bin_center, count, density) per PC: saved under {root_dir / key / 'histograms'}")
            lines.append(f"      - PNG plots (per PC) with histogram (density), KDE (Gaussian), and Gaussian fit: saved under {root_dir / key / 'plots'}")
        if cr.get("grid_plot"):
            lines.append("    Grid figure (all PCs):")
            lines.append(f"      - {cr['grid_plot']}")
            lines.append("        Meaning: Multi-panel PDF visualization across all retained PCs.")
        if cr.get("bin_compare_plots"):
            lines.append("    Bin-size comparison plots (per PC):")
            lines.append(f"      - PNG overlays comparing FD/Scott/Sturges and scaled-FD vs KDE: saved under {root_dir / key / 'plots_compare'}")
        lines.append("")
    lines.append("Notes:")
    lines.append("  - Outputs saved under a non-timestamped folder (Step3_Stats_Joint).")
    lines.append("  - KDE is a non-parametric density estimate; the plotted curve integrates to 1.")
    lines.append("  - Δc_i is intended for Step 5 coefficient perturbation magnitudes along each PC.")
    lines.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return log_path


# Orchestrator for one dataset
def process_one_dataset_joint(key: str,
                              scores: pd.DataFrame,
                              evr_full: pd.Series,
                              out_dir: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    hist_dir = out_dir / "histograms"
    plots_dir = out_dir / "plots"
    plots_compare_dir = out_dir / "plots_compare"

    stats_df = compute_per_pc_stats(scores)
    plan_df = compute_binning_plan(scores)

    summary_files = export_summaries_joint(stats_df, plan_df, evr_full, scores, out_dir)
    created_hist_and_plots = export_histograms_and_plots_joint(scores, plan_df, hist_dir, plots_dir)
    grid_plot_path = export_grid_plots_joint(scores, plan_df, plots_dir)

    # KDE bandwidth report
    kde_report = compute_kde_bandwidth_report(scores)
    kde_report_path = out_dir / "kde_bandwidth_report.csv"
    kde_report.to_csv(kde_report_path, index=True)
    summary_files["kde_bandwidth_report.csv"] = kde_report_path

    # Bin-size comparison figures
    bin_compare_plots = export_bin_size_comparison_plots(scores, out_dir, compare_width_multipliers=[1.0, 0.75, 0.5, 0.25])

    created_map = {
        "histograms_and_plots": created_hist_and_plots,
        "grid_plot": grid_plot_path,
        "bin_compare_plots": bin_compare_plots
    }
    return summary_files, created_map


# Main runner (callable from VSCode or notebooks; no CLI parsing)
def run_step3_joint_main():
    subdirs = ensure_output_dirs_joint(STEP3_JOINT_DIR, list(DATASETS.keys()))
    loaded = load_joint_scores_and_evr(STEP2_JOINT_DIR, DATASETS)

    all_summary_files: Dict[str, Dict[str, Path]] = {}
    all_created_files: Dict[str, Dict[str, List[Path]]] = {}

    for key in DATASETS.keys():
        print(f"Processing dataset: {key}")
        scores = loaded[key]["scores"]
        evr_full = loaded[key]["evr_full"]

        out_dir = subdirs[key]
        summary_files, created_files = process_one_dataset_joint(key, scores, evr_full, out_dir)
        all_summary_files[key] = summary_files
        all_created_files[key] = created_files

    context = {"step2_joint_dir": str(STEP2_JOINT_DIR)}
    log_path = write_inventory_log_joint(STEP3_JOINT_DIR, all_created_files, all_summary_files, context)
    print(f"Joint Step 3 completed. Inventory log written to: {log_path}")


# Entry point for VSCode "Run Python File"
if __name__ == "__main__":
    # Edit USER_PATHS above, then run
    run_step3_joint_main()