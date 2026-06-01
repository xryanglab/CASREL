#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy import stats as scipy_stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = "Regulator_contribution_summary.csv"
MIN_REGULATORS = 5
N_HEATMAP_EVENTS = 10
N_RANDOM_PAIRS = 50000
SEED = 42
PREFIX = "event_specificity"
LABEL_MAX_LEN = 30

np.random.seed(SEED)
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def shorten(s, maxlen=LABEL_MAX_LEN):
    return s if len(s) <= maxlen else s[:maxlen-2] + '..'

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df):,} rows")
print(f"  Unique AS events : {df['AS_Site'].nunique():,}")
print(f"  Unique RBPs      : {df['RBP'].nunique():,}")

df['Abs_Contribution'] = df['Contribution'].abs()

all_sets = {}
filtered_out = 0
for site, grp in df.groupby('AS_Site'):
    if len(grp) >= MIN_REGULATORS:
        all_sets[site] = set(grp['RBP'])
    else:
        filtered_out += 1

sites = sorted(all_sets.keys())
n_events = len(sites)
set_sizes = [len(all_sets[s]) for s in sites]
all_rbps = sorted(set().union(*all_sets.values())) if all_sets else []
n_pool = len(all_rbps)

print(f"\nQualified AS events (>= {MIN_REGULATORS} regulators): {n_events:,}")
print(f"Filtered out events (< {MIN_REGULATORS} regulators): {filtered_out:,}")
print(f"Unique RBPs in regulator pool: {n_pool}")

if n_events < 2:
    print(f"\nERROR: Need at least 2 qualified AS events. Exiting.")
    sys.exit(1)

if n_pool < MIN_REGULATORS:
    print(f"\nERROR: Total unique RBPs ({n_pool}) is less than MIN_REGULATORS ({MIN_REGULATORS}). Exiting.")
    sys.exit(1)

jac_mat = np.zeros((n_events, n_events))
jac_list = []
for i, j in combinations(range(n_events), 2):
    si, sj = all_sets[sites[i]], all_sets[sites[j]]
    jval = len(si & sj) / len(si | sj) if len(si | sj) > 0 else 0
    jac_mat[i, j] = jac_mat[j, i] = jval
    jac_list.append(jval)
np.fill_diagonal(jac_mat, 1.0)
jac_obs = np.array(jac_list)

size_array = np.array(set_sizes)
jac_rand = []
for _ in range(N_RANDOM_PAIRS):
    sz_a, sz_b = np.random.choice(size_array, 2, replace=True)
    sz_a = min(sz_a, n_pool)
    sz_b = min(sz_b, n_pool)
    a = set(np.random.choice(all_rbps, sz_a, replace=False))
    b = set(np.random.choice(all_rbps, sz_b, replace=False))
    jac_rand.append(len(a & b) / len(a | b) if len(a | b) > 0 else 0)
jac_rand = np.array(jac_rand)

mw_stat, mw_p = scipy_stats.mannwhitneyu(jac_obs, jac_rand, alternative='two-sided')
print(f"\n=== Jaccard Statistics ===")
print(f"  Observed median: {np.median(jac_obs):.4f}, mean: {np.mean(jac_obs):.4f}")
print(f"  Random   median: {np.median(jac_rand):.4f}, mean: {np.mean(jac_rand):.4f}")
print(f"  Mann-Whitney P = {mw_p:.2e}")

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 2, wspace=0.35)

ax_hist = fig.add_subplot(gs[0, 0])
bin_max = min(jac_obs.max() + 0.05, 1.0)
bins = np.arange(0, bin_max + 0.02, 0.02)
ax_hist.hist(jac_obs, bins=bins, color='#3C78B5', edgecolor='white', alpha=0.85, zorder=2)
ax_hist.axvline(np.median(jac_obs), color='#D62728', ls='--', lw=2,
                label=f'Median = {np.median(jac_obs):.3f}', zorder=3)
ax_hist.axvline(np.mean(jac_obs), color='#FF7F0E', ls=':', lw=2,
                label=f'Mean = {np.mean(jac_obs):.3f}', zorder=3)
ax_hist.set_xlabel('Pairwise Jaccard Index', fontsize=12)
ax_hist.set_ylabel('Frequency (AS event pairs)', fontsize=12)
ax_hist.set_title('Distribution of Regulator Overlap', fontsize=13, fontweight='bold')
ax_hist.legend(fontsize=9, loc='upper right', framealpha=0.9)

info_text = (f'Total pairs = {len(jac_obs):,}\n'
             f'Jaccard = 0 : {np.mean(jac_obs==0)*100:.1f}%\n'
             f'Jaccard < 0.1 : {np.mean(jac_obs<0.1)*100:.1f}%\n'
             f'Jaccard < 0.2 : {np.mean(jac_obs<0.2)*100:.1f}%\n'
             f'Median set size: {np.median(set_sizes):.0f} RBPs')
ax_hist.text(0.97, 0.72, info_text, transform=ax_hist.transAxes, fontsize=9,
             va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

ax_heat = fig.add_subplot(gs[0, 1])
n_hm = min(N_HEATMAP_EVENTS, n_events)
hm_idx = np.random.choice(n_events, n_hm, replace=False)
hm_idx = np.sort(hm_idx)
hm_sites = [sites[i] for i in hm_idx]
hm_mat = jac_mat[np.ix_(hm_idx, hm_idx)]
hm_labels = [shorten(s) for s in hm_sites]

dist_mat = 1.0 - hm_mat
np.fill_diagonal(dist_mat, 0.0)
condensed = squareform(dist_mat)
Z = linkage(condensed, method='average')
order = leaves_list(Z)
hm_mat_ordered = hm_mat[np.ix_(order, order)]
hm_labels_ordered = [hm_labels[i] for i in order]

im = ax_heat.imshow(hm_mat_ordered, cmap='YlOrRd', vmin=0, vmax=max(0.5, hm_mat_ordered.max()),
                    aspect='auto', interpolation='nearest')
ax_heat.set_xticks(range(n_hm))
ax_heat.set_xticklabels(hm_labels_ordered, rotation=90, fontsize=6)
ax_heat.set_yticks(range(n_hm))
ax_heat.set_yticklabels(hm_labels_ordered, fontsize=6)
ax_heat.set_title(f'Jaccard Similarity Heatmap\n({n_hm} sampled AS events, clustered)',
                  fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax_heat, shrink=0.7, pad=0.02)
cbar.set_label('Jaccard Index', fontsize=10)

plt.savefig(f'{PREFIX}_histogram_heatmap.pdf', bbox_inches='tight')
plt.savefig(f'{PREFIX}_histogram_heatmap.png', bbox_inches='tight')
print(f"\n>> Saved: {PREFIX}_histogram_heatmap.pdf/png")

stats = {
    'Total_qualified_AS_events': n_events,
    'Total_unique_RBPs_in_pool': n_pool,
    'Min_regulators_per_event': min(set_sizes),
    'Max_regulators_per_event': max(set_sizes),
    'Median_regulators_per_event': np.median(set_sizes),
    'Mean_regulators_per_event': np.mean(set_sizes),
    'Total_pairwise_comparisons': len(jac_obs),
    'Observed_median_Jaccard': np.median(jac_obs),
    'Observed_mean_Jaccard': np.mean(jac_obs),
    'Observed_std_Jaccard': np.std(jac_obs),
    'Random_median_Jaccard': np.median(jac_rand),
    'Random_mean_Jaccard': np.mean(jac_rand),
    'Mann_Whitney_P': mw_p,
    'Pct_Jaccard_equals_0': np.mean(jac_obs == 0) * 100,
    'Pct_Jaccard_below_0.1': np.mean(jac_obs < 0.1) * 100,
    'Pct_Jaccard_below_0.2': np.mean(jac_obs < 0.2) * 100,
}

stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
stats_df.to_csv(f'{PREFIX}_statistics.csv', index=False)
print(f"\n>> Saved: {PREFIX}_statistics.csv")
print("Done.")