#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def configure_plot_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.2,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })


def save_current_figure(output_dir: str, base_name: str) -> None:
    png_path = os.path.join(output_dir, f"{base_name}.png")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def load_detail(csv_path: str, stage_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["stage"] = stage_name
    return df


def load_summary(csv_path: str, stage_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["stage"] = stage_name
    return df


def save_latex_table(df: pd.DataFrame, out_path: str) -> None:
    cols = [
        "stage",
        "n_samples",
        "mae_x",
        "mae_y",
        "mae_norm",
        "rmse_x",
        "rmse_y",
        "rmse_norm",
        "final_error",
        "max_error",
    ]

    df2 = df[cols].copy()

    lines = []
    lines.append("\\begin{tabular}{lrrrrrrrrr}")
    lines.append("\\hline")
    lines.append(
        "stage & n\\_samples & mae\\_x & mae\\_y & mae\\_norm & rmse\\_x & rmse\\_y & rmse\\_norm & final\\_error & max\\_error \\\\"
    )
    lines.append("\\hline")

    for _, row in df2.iterrows():
        line = (
            f"{row['stage']} & "
            f"{int(row['n_samples'])} & "
            f"{float(row['mae_x']):.3f} & "
            f"{float(row['mae_y']):.3f} & "
            f"{float(row['mae_norm']):.3f} & "
            f"{float(row['rmse_x']):.3f} & "
            f"{float(row['rmse_y']):.3f} & "
            f"{float(row['rmse_norm']):.3f} & "
            f"{float(row['final_error']):.3f} & "
            f"{float(row['max_error']):.3f} \\\\"
        )
        lines.append(line)

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))




def make_summary_barplots(summary_df: pd.DataFrame, output_dir: str) -> None:
    metrics = [
        ("mae_norm", "MAE norm [m]"),
        ("rmse_norm", "RMSE norm [m]"),
        ("final_error", "Final error [m]"),
        ("max_error", "Max error [m]"),
    ]

    for metric, ylabel in metrics:
        plt.figure()
        plt.bar(summary_df["stage"], summary_df[metric])
        plt.ylabel(ylabel)
        plt.title(f"Confronto {metric}")
        save_current_figure(output_dir, f"bar_{metric}")


def make_trajectory_plot(detail_df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(8, 6))

    gt_done = False
    for stage, sdf in detail_df.groupby("stage", observed= False):
        sdf = sdf.sort_values("t")

        if not gt_done:
            plt.plot(
                sdf["gt_x"],
                sdf["gt_y"],
                label="GT relativa (GPS)",
            )
            gt_done = True

        plt.plot(
            sdf["est_x"],
            sdf["est_y"],
            label=f"Stima {stage}",
        )

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Confronto traiettorie")
    plt.legend()
    plt.axis("equal")
    save_current_figure(output_dir, "compare_trajectories")


def make_error_time_plot(detail_df: pd.DataFrame, output_dir: str) -> None:
    plt.figure()

    for stage, sdf in detail_df.groupby("stage", observed= False):
        sdf = sdf.sort_values("t").copy()
        sdf["t_rel"] = sdf["t"] - sdf["t"].iloc[0]
        plt.plot(
            sdf["t_rel"],
            sdf["err_norm"],
            label=stage,
        )

    plt.xlabel("t [s]")
    plt.ylabel("Errore euclideo [m]")
    plt.title("Errore nel tempo")
    plt.legend()
    save_current_figure(output_dir, "compare_error_time")


def make_boxplot(detail_df: pd.DataFrame, output_dir: str) -> None:
    stages = []
    data = []

    for stage, sdf in detail_df.groupby("stage", observed= False):
        stages.append(stage)
        data.append(sdf["err_norm"].values)

    plt.figure()
    plt.boxplot(data, tick_labels=stages)
    plt.ylabel("Errore euclideo [m]")
    plt.title("Distribuzione errore")
    save_current_figure(output_dir, "compare_error_boxplot")


def make_thesis_summary_csv(summary_df: pd.DataFrame, output_dir: str) -> None:
    cols = [
        "stage",
        "n_samples",
        "mae_x",
        "mae_y",
        "mae_norm",
        "rmse_x",
        "rmse_y",
        "rmse_norm",
        "final_error",
        "max_error",
    ]
    out_csv = os.path.join(output_dir, "thesis_summary_table.csv")
    summary_df[cols].to_csv(out_csv, index=False)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confronto metriche fusion/unicycle/ackermann con output da tesi"
    )
    parser.add_argument("--fusion-detail", required=True)
    parser.add_argument("--fusion-summary", required=True)
    parser.add_argument("--unicycle-detail", required=True)
    parser.add_argument("--unicycle-summary", required=True)
    parser.add_argument("--ackermann-detail", required=True)
    parser.add_argument("--ackermann-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style()

    fusion_detail = load_detail(args.fusion_detail, "fusion")
    fusion_summary = load_summary(args.fusion_summary, "fusion")

    unicycle_detail = load_detail(args.unicycle_detail, "unicycle")
    unicycle_summary = load_summary(args.unicycle_summary, "unicycle")

    ackermann_detail = load_detail(args.ackermann_detail, "ackermann")
    ackermann_summary = load_summary(args.ackermann_summary, "ackermann")

    detail_df = pd.concat(
        [fusion_detail, unicycle_detail, ackermann_detail],
        ignore_index=True,
    )
    summary_df = pd.concat(
        [fusion_summary, unicycle_summary, ackermann_summary],
        ignore_index=True,
    )

    stage_order = ["fusion", "unicycle", "ackermann"]

    summary_df["stage"] = pd.Categorical(
        summary_df["stage"], categories=stage_order, ordered=True
    )
    summary_df = summary_df.sort_values("stage").reset_index(drop=True)

    detail_df["stage"] = pd.Categorical(
        detail_df["stage"], categories=stage_order, ordered=True
    )
    detail_df = detail_df.sort_values(["stage", "t"]).reset_index(drop=True)

    make_thesis_summary_csv(summary_df, str(output_dir))
    save_latex_table(summary_df, os.path.join(output_dir, "thesis_summary_table.tex"))

    make_trajectory_plot(detail_df, str(output_dir))
    make_error_time_plot(detail_df, str(output_dir))
    make_boxplot(detail_df, str(output_dir))
    make_summary_barplots(summary_df, str(output_dir))

    print(f"Output salvati in: {output_dir}")


if __name__ == "__main__":
    main()


