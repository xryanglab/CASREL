#!/usr/bin/env python

import argparse
import os
import re
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from scipy.stats import mannwhitneyu

plt.rcParams.update({
    "font.family": "Arial",
    "font.size":   10,
    "figure.dpi":  300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PALETTE = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
           "#8491B4", "#91D1C2", "#DC0000"]


def load_gene_annotation(gtf_path, upper_gene_name=False):
    df = pd.read_csv(gtf_path, sep="\t")
    if upper_gene_name:
        df["gene_name"] = df["gene_name"].str.upper()
    genes = {}
    for chromo, sub in df.groupby("chr"):
        agg = (sub.groupby("gene_name")
                  .agg(chr=("chr", "max"),
                       start=("start", "min"),
                       end=("end", "max"))
                  .sort_values("start"))
        genes[chromo] = agg[["start", "end"]]
    return genes


def _range_distance(x, y, a, b):
    if x > a:
        x, y, a, b = a, b, x, y
    return a - min(y, b)


def _find_closest_gene(site, genes_dict):
    parts = site.split("_")
    if len(parts) != 3:
        return "unknown"
    chromo, start, end = parts
    start = start.split(".")[0]
    end   = end.split(".")[0]
    try:
        start, end = int(start), int(end)
    except ValueError:
        return "unknown"
    if chromo not in genes_dict:
        return "unknown"
    cgenes = genes_dict[chromo].copy()
    cgenes["dist"] = cgenes.apply(
        lambda r: _range_distance(r["start"], r["end"], start, end), axis=1)
    return cgenes.sort_values("dist").index[0]


def annotate_gene_column(df, genes_dict):
    print("  Annotating Gene column via coordinate look-up ...")
    unique_sites = df["AS_Site"].unique()
    print(f"  Looking up {len(unique_sites)} unique AS sites ...")
    site_map = {s: _find_closest_gene(s, genes_dict) for s in unique_sites}
    df = df.copy()
    df["Gene"] = df["AS_Site"].map(site_map)
    before = len(df)
    df = df[df["Gene"] != "unknown"].reset_index(drop=True)
    print(f"  Gene annotation complete: kept {len(df)}/{before} rows "
          f"({before - len(df)} removed as 'unknown').")
    return df


def _extract_gene_fallback(site_name):
    clean = re.sub(r"_[se]$", "", str(site_name))
    for p in clean.split(":"):
        p = p.strip()
        if p and not p.startswith("chr") and not p.replace("-", "").replace("+", "").isdigit():
            return p
    for p in clean.split("_"):
        p = p.strip()
        if p and not p.startswith("chr") and not p.isdigit() and len(p) > 1:
            return p
    return clean


def load_reg_summary(path, label=None, genes_dict=None):
    df = pd.read_csv(path)
    required = {"RBP", "AS_Site", "Contribution", "Direction"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Missing columns in {path}. Need: {required}")

    if "Gene" not in df.columns:
        if genes_dict is not None:
            df = annotate_gene_column(df, genes_dict)
        else:
            print("  [WARN] No 'Gene' column and no annotation file provided; "
                  "using text-based gene extraction (may be inaccurate).")
            df["Gene"] = df["AS_Site"].apply(_extract_gene_fallback)

    if label:
        df["Dataset"] = label
    return df


def _overlap_coef(a: set, b: set) -> float:
    denom = min(len(a), len(b))
    return len(a & b) / denom if denom > 0 else 0.0


def compute_overlap_pairs(df, top_n=None):
    site_rbps: dict = {}
    site_gene: dict = {}
    for site, grp in df.groupby("AS_Site"):
        if top_n:
            grp = grp.nlargest(top_n, "Contribution")
        rbp_set = set(grp["RBP"].unique())
        if rbp_set:
            site_rbps[site] = rbp_set
            gene_vals = grp["Gene"].unique()
            site_gene[site] = gene_vals[0] if len(gene_vals) > 0 else "unknown"

    gene_sites: dict = defaultdict(list)
    for site, gene in site_gene.items():
        gene_sites[gene].append(site)
    multi_genes = {g: s for g, s in gene_sites.items() if len(s) >= 2}

    within_ovlps = []
    gene_stats_rows = []
    for gene, sites in multi_genes.items():
        po = []
        for s1, s2 in combinations(sites, 2):
            if s1 in site_rbps and s2 in site_rbps:
                ovlp = _overlap_coef(site_rbps[s1], site_rbps[s2])
                within_ovlps.append(ovlp)
                po.append(ovlp)
        if po:
            gene_stats_rows.append(dict(
                gene=gene,
                n_events=len(sites),
                mean_overlap=float(np.mean(po)),
                max_overlap=float(np.max(po)),
            ))

    all_sites = list(site_rbps.keys())
    rng = np.random.default_rng(42)
    n_between = min(len(within_ovlps) * 5, 50_000)
    between_ovlps = []
    attempts = 0
    while len(between_ovlps) < n_between and attempts < n_between * 10:
        attempts += 1
        i, j = rng.choice(len(all_sites), size=2, replace=False)
        s1, s2 = all_sites[i], all_sites[j]
        if site_gene.get(s1) != site_gene.get(s2):
            between_ovlps.append(_overlap_coef(site_rbps[s1], site_rbps[s2]))

    gene_stats = pd.DataFrame(gene_stats_rows)
    return within_ovlps, between_ovlps, gene_stats


def plot_event_count_histogram(gene_stats, dataset_label, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    if not gene_stats.empty:
        nev = gene_stats["n_events"].values
        ax.hist(nev, bins=range(2, int(nev.max()) + 2),
                color=PALETTE[0], alpha=0.75, edgecolor="white",
                linewidth=0.5, zorder=3, align="left")
        ax.set_xlabel("Number of AS Events per Gene")
        ax.set_ylabel("Number of Genes")
        ax.set_title(f"Multi-Event Genes (n = {len(gene_stats)} genes with >=2 events)",
                     fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(output_dir, f"{dataset_label}_event_count_distribution.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_overlap_violin(within_ovlps, between_ovlps, dataset_label, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 6))
    colors = [PALETTE[2], PALETTE[3]]
    data_w = within_ovlps
    data_b = between_ovlps
    if data_w and data_b:
        parts = ax.violinplot([data_w, data_b], positions=[0, 1], showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i]); pc.set_alpha(0.4)
        bp = ax.boxplot([data_w, data_b], positions=[0, 1], widths=0.3,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=2))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i]); patch.set_alpha(0.75)
        for i, d in enumerate([data_w, data_b]):
            med = float(np.median(d))
            ax.text(i, med + 0.03, f"med={med:.3f}",
                    ha="center", fontsize=9, fontweight="bold", color=colors[i])
        if len(data_w) >= 3 and len(data_b) >= 3:
            _, pval = mannwhitneyu(data_w, data_b, alternative="greater")
            sig = ("***" if pval < 0.001 else
                   ("**"  if pval < 0.01  else
                    ("*"   if pval < 0.05  else "n.s.")))
            ymax = max(np.percentile(data_w, 95), np.percentile(data_b, 95))
            ax.plot([0, 0, 1, 1],
                    [ymax + 0.05, ymax + 0.08, ymax + 0.08, ymax + 0.05],
                    "k-", linewidth=1.2)
            ax.text(0.5, ymax + 0.09, f"{sig}\np={pval:.2e}",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Within-gene\n(n={len(data_w)})",
         f"Between-gene\n(n={len(data_b)})"],
        fontsize=9)
    ax.set_ylabel("Overlap Coefficient", fontsize=9)
    ax.set_title("RBP Sharing: Within-Gene vs Between-Gene\n(Overlap Coefficient)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(output_dir, f"{dataset_label}_overlap_coefficient_violin.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate two plots: event count histogram and overlap coefficient violin.")
    parser.add_argument("--input", required=True,
                        help="Path to Regulator_contribution_summary.csv file")
    parser.add_argument("--output_dir", default="coordinated_splicing_results")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Use only top-N RBPs per site (by Contribution)")
    parser.add_argument("--gtf_file", type=str, default=None,
                        help="Gene annotation file (tab-separated: chr, start, end, gene_name).")
    parser.add_argument("--species", type=str, choices=["human", "mouse"], default="human")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gtf_path = args.gtf_file
    if gtf_path is None:
        gtf_path = ("./data/hg38_protein_gtf.txt" if args.species == "human" else
                    "./data/mm10_pro_gtf.txt")

    genes_dict = None
    if os.path.exists(gtf_path):
        upper = (args.species == "mouse")
        print(f"Loading gene annotation from: {gtf_path}")
        genes_dict = load_gene_annotation(gtf_path, upper_gene_name=upper)
        chrs_preview = sorted(genes_dict.keys())[:8]
        print(f"  Chromosomes loaded (first 8 shown): {chrs_preview}")
    else:
        print(f"[WARN] Annotation file not found at '{gtf_path}'.\n"
              f"       Text-based gene extraction will be used as fallback.\n"
              f"       Pass --gtf_file to enable accurate gene mapping.")

    dataset_label = os.path.splitext(os.path.basename(args.input))[0]
    print(f"\nProcessing: {dataset_label}")
    df = load_reg_summary(args.input, dataset_label, genes_dict)
    print(f"  Loaded: {len(df)} rows | {df['Gene'].nunique()} genes | "
          f"{df['AS_Site'].nunique()} AS sites | {df['RBP'].nunique()} RBPs")

    within_ovlps, between_ovlps, gene_stats = compute_overlap_pairs(df, top_n=args.top_n)

    def _med(lst):
        return f"{np.median(lst):.3f}" if lst else "N/A"
    print(f"  Within-gene pairs : {len(within_ovlps):>7} | median Overlap = {_med(within_ovlps)}")
    print(f"  Between-gene pairs: {len(between_ovlps):>7} | median Overlap = {_med(between_ovlps)}")

    if within_ovlps and between_ovlps:
        _, pval = mannwhitneyu(within_ovlps, between_ovlps, alternative="greater")
        print(f"  Mann-Whitney U (within > between) : Overlap p = {pval:.2e}")

    plot_event_count_histogram(gene_stats, dataset_label, args.output_dir)
    plot_overlap_violin(within_ovlps, between_ovlps, dataset_label, args.output_dir)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()