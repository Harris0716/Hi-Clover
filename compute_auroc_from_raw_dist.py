#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def load_threshold(value):
    value = np.asarray(value)
    return float(value.reshape(-1)[0])


def best_mean_threshold(dist, lbl, steps=5000):
    """
    lbl == 0: replicate
    lbl == 1: condition

    Prediction rule:
      distance < threshold  -> replicate
      distance >= threshold -> condition
    """
    dist = np.asarray(dist).reshape(-1)
    lbl = np.asarray(lbl).reshape(-1)

    rep = dist[lbl == 0]
    cond = dist[lbl == 1]

    d_min, d_max = float(dist.min()), float(dist.max())
    eps = max((d_max - d_min) * 1e-6, 1e-8)
    thresholds = np.linspace(d_min - eps, d_max + eps, steps)

    rep_acc = (rep[:, None] < thresholds[None, :]).mean(axis=0)
    cond_acc = (cond[:, None] >= thresholds[None, :]).mean(axis=0)
    mean_perf = (rep_acc + cond_acc) / 2.0

    idx = int(np.argmax(mean_perf))
    return {
        "best_threshold": float(thresholds[idx]),
        "rep_acc_at_best": float(rep_acc[idx]),
        "cond_acc_at_best": float(cond_acc[idx]),
        "mean_perf_at_best": float(mean_perf[idx]),
        "sep_index_at_best": float(2 * mean_perf[idx] - 1),
    }


def compute_one(path, dataset, phase, threshold_steps=5000):
    data = np.load(path)
    dist = np.asarray(data["dist"]).reshape(-1)
    lbl = np.asarray(data["lbl"]).reshape(-1)

    # Important:
    # lbl == 0 means replicate, lbl == 1 means condition.
    # Larger distance means more likely condition.
    # Therefore, AUROC should use score = dist.
    auroc = roc_auc_score(lbl, dist)
    auprc = average_precision_score(lbl, dist)

    out = {
        "dataset": dataset,
        "phase": phase,
        "file": path,
        "n_pairs": int(len(lbl)),
        "n_replicate_pairs": int(np.sum(lbl == 0)),
        "n_condition_pairs": int(np.sum(lbl == 1)),
        "replicate_mean_distance": float(dist[lbl == 0].mean()),
        "condition_mean_distance": float(dist[lbl == 1].mean()),
        "auroc": float(auroc),
        "auprc_condition": float(auprc),
    }

    if "threshold" in data:
        out["stored_threshold"] = load_threshold(data["threshold"])

    out.update(best_mean_threshold(dist, lbl, steps=threshold_steps))
    return out, dist, lbl


def plot_roc(dist, lbl, title, out_path):
    fpr, tpr, _ = roc_curve(lbl, dist)
    auc = roc_auc_score(lbl, dist)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, linewidth=2.0, label=f"AUROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pr(dist, lbl, title, out_path):
    precision, recall, _ = precision_recall_curve(lbl, dist)
    auprc = average_precision_score(lbl, dist)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(recall, precision, linewidth=2.0, label=f"AUPRC = {auprc:.4f}")
    baseline = np.mean(lbl == 1)
    ax.axhline(baseline, linestyle="--", linewidth=1.0, color="gray", label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute AUROC/AUPRC from *_raw_dist.npz files.")
    parser.add_argument("--data_dir", required=True, help="Directory containing *_raw_dist.npz files")
    parser.add_argument("--out_dir", default="auroc_results", help="Output directory")
    parser.add_argument("--datasets", default="liver,NPC,TCell", help="Comma-separated dataset prefixes")
    parser.add_argument("--phases", default="train_val,test", help="Comma-separated phases")
    parser.add_argument("--threshold_steps", type=int, default=5000)
    parser.add_argument("--plot", action="store_true", help="Save ROC and PR curve PDFs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    phases = [x.strip() for x in args.phases.split(",") if x.strip()]

    rows = []

    for dataset in datasets:
        for phase in phases:
            path = os.path.join(args.data_dir, f"{dataset}_{phase}_raw_dist.npz")
            if not os.path.exists(path):
                print(f"[Missing] {path}")
                continue

            print(f"[Read] {path}")
            row, dist, lbl = compute_one(path, dataset, phase, threshold_steps=args.threshold_steps)
            rows.append(row)

            if args.plot:
                safe_name = f"{dataset}_{phase}"
                plot_roc(
                    dist,
                    lbl,
                    title=f"ROC Curve: {dataset} ({phase})",
                    out_path=os.path.join(args.out_dir, f"{safe_name}_roc.pdf"),
                )
                plot_pr(
                    dist,
                    lbl,
                    title=f"Precision-Recall Curve: {dataset} ({phase})",
                    out_path=os.path.join(args.out_dir, f"{safe_name}_pr.pdf"),
                )

    if not rows:
        raise FileNotFoundError(
            "No *_raw_dist.npz files were found. Check --data_dir and file names."
        )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "auroc_summary.csv")
    df.to_csv(out_csv, index=False)

    print("\n========== AUROC Summary ==========")
    display_cols = [
        "dataset", "phase", "auroc", "auprc_condition",
        "mean_perf_at_best", "sep_index_at_best",
        "best_threshold", "replicate_mean_distance", "condition_mean_distance",
        "n_pairs",
    ]
    print(df[[c for c in display_cols if c in df.columns]].to_string(index=False))
    print(f"\nSaved: {out_csv}")

    if args.plot:
        print(f"Saved ROC/PR plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
