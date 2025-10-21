#!/usr/bin/env python3
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_df(path, label):
    df = pd.read_csv(path)
    df["p_sum"] = df["pX"] + df["pY"] + df["pZ"]
    df["dataset"] = label
    return df

def ecdf(a):
    x = np.sort(a)
    y = (np.arange(1, len(x)+1))/len(x)
    return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-a", required=True, help="First CSV (e.g., out/batch_compare_nn.csv)")
    ap.add_argument("--label-a", default="Adaptive (NN)")
    ap.add_argument("--csv-b", help="Optional second CSV (e.g., out/batch_compare_direct.csv)")
    ap.add_argument("--label-b", default="Adaptive (Direct)")
    ap.add_argument("--outdir", default="out/viz")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dfa = load_df(args.csv_a, args.label_a)
    dfs = [dfa]
    if args.csv_b:
        dfb = load_df(args.csv_b, args.label_b)
        dfs.append(dfb)
    df = pd.concat(dfs, ignore_index=True)

    # Summary CSV
    summary = df.groupby("dataset").agg(
        n=("score_improvement","size"),
        win_frac=("score_improvement", lambda s: (s>0).mean()),
        avg_improv=("score_improvement","mean"),
        med_improv=("score_improvement","median")
    ).reset_index()
    summary.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)

    # 1) Histogram(s)
    for label, sub in df.groupby("dataset"):
        plt.figure()
        sub["score_improvement"].hist(bins=60)
        plt.xlabel("Adaptive − Static (score)")
        plt.ylabel("Count")
        plt.title(f"Score Improvement Distribution — {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"hist_{label.replace(' ','_')}.png"))

    # 2) ECDF comparison
    plt.figure()
    for label, sub in df.groupby("dataset"):
        x, y = ecdf(sub["score_improvement"].values)
        plt.plot(x, y, drawstyle="steps-post", label=label)
    plt.xlabel("Adaptive − Static (score)")
    plt.ylabel("ECDF")
    plt.title("ECDF of Score Improvement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "ecdf_compare.png"))

    # 3) Scatter: improvement vs total noise (per dataset)
    for label, sub in df.groupby("dataset"):
        plt.figure()
        plt.scatter(sub["p_sum"], sub["score_improvement"], s=4, alpha=0.5)
        plt.xlabel("pX + pY + pZ")
        plt.ylabel("Adaptive − Static (score)")
        plt.title(f"Improvement vs Total Noise — {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"scatter_total_noise_{label.replace(' ','_')}.png"))

    # 4) Win rate by noise bins
    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30]
    labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
    df["p_bin"] = pd.cut(df["p_sum"], bins=bins, labels=labels, include_lowest=True)
    win_rate = df.groupby(["dataset","p_bin"])["score_improvement"].apply(lambda s:(s>0).mean()).unstack(0)
    win_rate.to_csv(os.path.join(args.outdir, "win_rate_by_bin.csv"))

    plt.figure()
    win_rate.plot(kind="bar")
    plt.ylabel("Adaptive win fraction")
    plt.title("Adaptive Wins by Total Noise Bin")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "win_rate_by_bin.png"))

    # 5) Boxplot of improvements by decision_source (lookup/nn/direct_opt)
    plt.figure()
    df.boxplot(column="score_improvement", by=["dataset","source"], vert=True, rot=30)
    plt.suptitle("")
    plt.title("Score Improvement by Decision Source")
    plt.xlabel("Dataset, Source")
    plt.ylabel("Adaptive − Static (score)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "box_improvement_by_source.png"))

    # 6) Per-method head-to-head (optional): which adaptive schedules win more?
    per_method = df.groupby(["dataset","adapt_method"]).agg(
        n=("score_improvement","size"),
        win_frac=("score_improvement", lambda s:(s>0).mean()),
        avg_improv=("score_improvement","mean")
    ).sort_values(["dataset","win_frac","avg_improv"], ascending=[True,False,False])
    per_method.to_csv(os.path.join(args.outdir, "per_adaptive_method.csv"))

    print("Saved:")
    print(" -", os.path.join(args.outdir, "summary.csv"))
    print(" -", os.path.join(args.outdir, "ecdf_compare.png"))
    print(" -", os.path.join(args.outdir, "win_rate_by_bin.png"))
    print(" -", os.path.join(args.outdir, "box_improvement_by_source.png"))
    print(" -", os.path.join(args.outdir, "per_adaptive_method.csv"))
    print(" - histograms and scatter plots per dataset in", args.outdir)

if __name__ == "__main__":
    main()
