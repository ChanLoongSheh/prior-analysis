# Function Block [Module 2: Statistical Analysis] — Step 3 (Seaborn-free)
# Title: Statistical Analysis of PCA Coefficients for Ground-based Atmospheric Profiles
# Author: (Your Lab)
# Description:
#   Implements Step 3 (README.md) using only NumPy, Pandas, SciPy, and Matplotlib.
#   For each retained PC set (T_k95, T_k99, Td_k95, Td_k99):
#     - Load scores C (n_samples × k) and full explained variance ratios (EVR).
#     - Compute descriptive stats per PC.
#     - Compute histogram binning plan (Freedman–Diaconis, Scott, Sturges) and select recommended bins/width.
#     - Export per-PC histogram CSVs and PDF plots (histogram density + Gaussian fit + KDE via SciPy).
#     - Export grid figure for quick visual screening.
#     - Export summary CSVs: per-PC stats, binning plan, Δc_i (recommended bin width), EVR for retained k.
#     - Write a detailed inventory log with file paths and meanings.
#
# Notation:
#   - C (scores): principal component coefficients matrix.
#   - Δc_i (delta_c for PC i): recommended coefficient differential = selected histogram bin width.
#
# How to call (Main Program):
#   - Edit USER_PATHS (STEP2_DIR, STEP3_DIR) if needed.
#   - Run this script in Python. No command-line parsing is used.
#
# Outputs:
#   - Step3_Stats/[T_k95|T_k99|Td_k95|Td_k99]/
#       per_PC_stats.csv
#       per_PC_binning_plan.csv
#       delta_c_recommended.csv
#       explained_variance_ratio_k.csv
#       histograms/PCi_histogram.csv
#       plots/PCi_pdf.png
#       plots/all_PCs_pdf_grid.png
#   - Step3_Stats/Step3_Stats_inventory.log

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # batch/non-interactive
import matplotlib.pyplot as plt

# ------------------------
# USER_PATHS (edit here)
# ------------------------
STEP2_DIR = Path("./Step2_PCA")
STEP3_DIR = STEP2_DIR.parent / "Step3_Stats"

DATASETS = {
    "T_k95": ("PCA_T_scores_k95.csv", "PCA_T_explained_variance_ratio.csv"),
    "T_k99": ("PCA_T_scores_k99.csv", "PCA_T_explained_variance_ratio.csv"),
    "Td_k95": ("PCA_Td_scores_k95.csv", "PCA_Td_explained_variance_ratio.csv"),
    "Td_k99": ("PCA_Td_scores_k99.csv", "PCA_Td_explained_variance_ratio.csv"),
}

# ------------------------
# PLOTTING STYLE (no seaborn)
# ------------------------
plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 140
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


# Function 1 — Title: Safe creation of output directories
def ensure_output_dirs(root: Path, dataset_keys: List[str]) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    subdirs = {}
    for k in dataset_keys:
        d = root / k
        d.mkdir(parents=True, exist_ok=True)
        (d / "histograms").mkdir(parents=True, exist_ok=True)
        (d / "plots").mkdir(parents=True, exist_ok=True)
        subdirs[k] = d
    return subdirs


# Function 2 — Title: Load PCA scores and explained-variance ratios
def load_scores_and_evr(step2_dir: Path,
                        datasets: Dict[str, Tuple[str, str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    result = {}
    for key, (scores_csv, evr_csv) in datasets.items():
        scores_path = step2_dir / scores_csv
        evr_path = step2_dir / evr_csv

        if not scores_path.is_file():
            raise FileNotFoundError(f"Scores file not found for {key}: {scores_path}")
        if not evr_path.is_file():
            raise FileNotFoundError(f"Explained variance ratio file not found for {key}: {evr_path}")

        scores = pd.read_csv(scores_path, index_col=0)
        evr_df = pd.read_csv(evr_path, index_col=0, header=0)

        # Normalize EVR to a Series
        if evr_df.shape[1] == 1:
            evr_series = evr_df.iloc[:, 0]
        else:
            if "explained_variance_ratio" in evr_df.columns:
                evr_series = evr_df["explained_variance_ratio"]
            else:
                evr_series = evr_df.iloc[:, 0]

        result[key] = {"scores": scores, "evr_full": evr_series}
    return result


# Function 3 — Title: Compute descriptive statistics per PC
def compute_per_pc_stats(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in scores.columns:
        x = scores[col].to_numpy()
        n = x.size
        mean = float(np.mean(x))
        std = float(np.std(x, ddof=1)) if n > 1 else np.nan
        skew = float(stats.skew(x, bias=False)) if n > 2 else np.nan
        kurt = float(stats.kurtosis(x, fisher=True, bias=False)) if n > 3 else np.nan
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


# Function 4 — Title: Compute histogram binning plan (FD, Scott, Sturges)
def compute_binning_plan(scores: pd.DataFrame,
                         min_bins: int = 5,
                         max_bins: int = 200) -> pd.DataFrame:
    plan_rows = []
    for col in scores.columns:
        x = scores[col].to_numpy()
        n = x.size
        xmin, xmax = np.min(x), np.max(x)
        data_range = float(xmax - xmin) if xmax > xmin else 1.0
        q25, q75 = np.percentile(x, [25, 75])
        iqr = float(q75 - q25)
        std = float(np.std(x, ddof=1)) if n > 1 else 0.0

        # Freedman–Diaconis
        fd_width = 2.0 * iqr * (n ** (-1.0/3.0)) if iqr > 0 else np.nan
        if np.isfinite(fd_width) and fd_width > 0:
            fd_bins = int(np.ceil(data_range / fd_width))
        else:
            fd_bins = np.nan

        # Scott
        scott_width = 3.5 * std * (n ** (-1.0/3.0)) if std > 0 else np.nan
        if np.isfinite(scott_width) and scott_width > 0:
            scott_bins = int(np.ceil(data_range / scott_width))
        else:
            scott_bins = np.nan

        # Sturges
        sturges_bins = int(np.ceil(np.log2(n) + 1)) if n > 0 else 1

        # Choose recommendation with clamping
        recommended_rule = None
        recommended_bins = None
        recommended_width = None

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
            "data_min": float(xmin),
            "data_max": float(xmax),
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


# Function 5 — Title: Export histograms (CSV) and per-PC plots (PNG, KDE via SciPy)
def export_histograms_and_plots(scores: pd.DataFrame,
                                bin_plan: pd.DataFrame,
                                out_hist_dir: Path,
                                out_plots_dir: Path) -> List[Path]:
    created_files: List[Path] = []
    for col in scores.columns:
        x = scores[col].to_numpy()
        rec_bins = int(bin_plan.loc[col, "recommended_bins"])
        xmin = float(bin_plan.loc[col, "data_min"])
        xmax = float(bin_plan.loc[col, "data_max"])

        # For degenerate ranges, expand slightly
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

        # Plot: histogram (density), KDE (scipy), Gaussian fit
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.hist(x, bins=edges, density=True, alpha=0.35, color="tab:blue",
                edgecolor="black", linewidth=0.5, label="Histogram (density)")

        # KDE via SciPy if variance > 0
        sigma = np.std(x, ddof=1) if x.size > 1 else 0.0
        if sigma > 0:
            kde = stats.gaussian_kde(x)
            grid = np.linspace(edges[0], edges[-1], 300)
            kde_y = kde(grid)
            ax.plot(grid, kde_y, color="tab:orange", linewidth=1.5, label="KDE (SciPy)")

        # Gaussian fit overlay using sample mean and std
        mu = float(np.mean(x))
        if sigma > 0:
            grid = np.linspace(edges[0], edges[-1], 300)
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


# Function 6 — Title: Export grid figures for all PCs in a dataset
def export_grid_plots(scores: pd.DataFrame,
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

    # Turn off unused panels
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle("Coefficient PDFs (all PCs)", y=1.02, fontsize=13)
    fig.tight_layout()
    out_path = out_plots_dir / "all_PCs_pdf_grid.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# Function 7 — Title: Save summary CSVs (stats, bin plan, Δc_i, EVR_k)
def export_summaries(stats_df: pd.DataFrame,
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

    # EVR_k: match retained k
    k = scores_df.shape[1]
    evr_k = evr_full.iloc[:k].copy()
    evr_k.index = [f"PC{i+1}" for i in range(k)]
    evr_k.name = "explained_variance_ratio"
    evr_k_csv = out_dir / "explained_variance_ratio_k.csv"
    evr_k.to_csv(evr_k_csv, header=True, index=True)
    files["explained_variance_ratio_k.csv"] = evr_k_csv

    return files


# Function 8 — Title: Write inventory log with detailed meanings
def write_inventory_log(root_dir: Path,
                        created_map: Dict[str, Dict[str, List[Path]]],
                        summary_map: Dict[str, Dict[str, Path]],
                        context: Dict[str, str]) -> Path:
    log_path = root_dir / "Step3_Stats_inventory.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("Ground-based Atmospheric Profiles — Step 3: Statistical Analysis of PCA Coefficients")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("")
    lines.append("Inputs (from Step 2):")
    lines.append(f"  Step 2 directory: {context.get('step2_dir','')}")
    lines.append("  - PCA_T_scores_k95.csv / PCA_T_scores_k99.csv: Truncated temperature scores C = X_centered · W (rows: samples, columns: PCs).")
    lines.append("  - PCA_Td_scores_k95.csv / PCA_Td_scores_k99.csv: Truncated dew-point scores C = X_centered · W (rows: samples, columns: PCs).")
    lines.append("  - PCA_T_explained_variance_ratio.csv / PCA_Td_explained_variance_ratio.csv: Full explained variance ratios per PC (sklearn definition).")
    lines.append("")
    lines.append("Core quantities:")
    lines.append("  - C (scores): Principal component coefficients matrix for the retained PCs.")
    lines.append("  - Δc_i (delta_c): Recommended coefficient differential for PC i, chosen as the histogram bin width from the selected rule.")
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
                lines.append("        Meaning: Per-PC histogram binning plan (FD/Scott/Sturges), recommended rule, bins, and bin width.")
            if "delta_c_recommended.csv" in sm:
                lines.append(f"      - {sm['delta_c_recommended.csv']}")
                lines.append("        Meaning: Recommended Δc_i per PC (equal to recommended histogram bin width).")
            if "explained_variance_ratio_k.csv" in sm:
                lines.append(f"      - {sm['explained_variance_ratio_k.csv']}")
                lines.append("        Meaning: Explained variance ratios for the retained PCs (truncated to k).")
        created = created_map[key]
        hlist = created.get("histograms_and_plots", [])
        if hlist:
            lines.append("    Histograms and per-PC plots:")
            lines.append(f"      - CSV histograms (bin_left, bin_right, bin_center, count, density) for each PC: saved under {root_dir / key / 'histograms'}")
            lines.append(f"      - PNG plots (per PC) with histogram (density), KDE (SciPy), and Gaussian fit: saved under {root_dir / key / 'plots'}")
        grid_path = created.get("grid_plot")
        if grid_path:
            lines.append(f"    Grid figure (all PCs):")
            lines.append(f"      - {grid_path}")
            lines.append("        Meaning: Multi-panel PDF visualization across all retained PCs.")
        lines.append("")

    lines.append("Notes:")
    lines.append("  - No detector system effects, thermal emission, or noise considerations are included in this stage.")
    lines.append("  - Row indices from Step 2 scores are preserved where applicable.")
    lines.append("  - Δc_i is intended for Step 5 profile perturbation magnitudes along each PC.")
    lines.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return log_path


# Function 9 — Title: Orchestrator for one dataset (T_k95, T_k99, Td_k95, Td_k99)
def process_one_dataset(key: str,
                        scores: pd.DataFrame,
                        evr_full: pd.Series,
                        out_dir: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    hist_dir = out_dir / "histograms"
    plots_dir = out_dir / "plots"

    stats_df = compute_per_pc_stats(scores)
    plan_df = compute_binning_plan(scores)
    summary_files = export_summaries(stats_df, plan_df, evr_full, scores, out_dir)
    created_files_list = export_histograms_and_plots(scores, plan_df, hist_dir, plots_dir)
    grid_plot_path = export_grid_plots(scores, plan_df, plots_dir)

    created_map = {
        "histograms_and_plots": created_files_list,
        "grid_plot": grid_plot_path
    }
    return summary_files, created_map


# Main Program — Title: Step 3 runner (no command-line parsing)
def main():
    subdirs = ensure_output_dirs(STEP3_DIR, list(DATASETS.keys()))
    loaded = load_scores_and_evr(STEP2_DIR, DATASETS)

    all_summary_files: Dict[str, Dict[str, Path]] = {}
    all_created_files: Dict[str, Dict[str, List[Path]]] = {}

    for key in DATASETS.keys():
        print(f"Processing dataset: {key}")
        scores = loaded[key]["scores"]
        evr_full = loaded[key]["evr_full"]

        out_dir = subdirs[key]
        summary_files, created_files = process_one_dataset(key, scores, evr_full, out_dir)
        all_summary_files[key] = summary_files
        all_created_files[key] = created_files

    context = {"step2_dir": str(STEP2_DIR)}
    log_path = write_inventory_log(STEP3_DIR, all_created_files, all_summary_files, context)
    print(f"Step 3 completed. Inventory log written to: {log_path}")


if __name__ == "__main__":
    main()