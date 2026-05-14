# CASREL: Cell-specific alternative splicing regulation inference via explainable learning
CASREL is a machine learning framework that infers RBP–alternative splicing regulatory networks de novo from single-cell RNA sequencing data, free of prior RBP–RNA binding annotations. By leveraging distinctive properties of single-cell splicing profiles and integrating ensemble learning with SHAP-based interpretation, CASREL enables accurate, interpretable, and cell-specific mapping of splicing regulation in physiological contexts.


## 🚀 Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xryanglab/CASREL/blob/main/demo/CASREL_demo_notebook.ipynb)

The `demo/` directory contains an example notebook and a minimal test dataset
to help users get started with CASREL. You can open and run
[`CASREL_demo_notebook.ipynb`](https://colab.research.google.com/github/xryanglab/CASREL/blob/main/demo/CASREL_demo_notebook.ipynb)
directly in Google Colab — the entire process takes approximately 10–15 minutes.
These materials demonstrate the complete workflow using a simplified dataset
while serving as templates for analyzing your own data.


## Tutorial ##

### Environment Setup
```bash
> conda create -n casrel -c conda-forge python=3.10.12 -y
> source activate casrel
> pip install -r requirements.txt

# Optional but not recommended
> conda install -c conda-forge numpy pandas scikit-learn lightgbm easydict -y
> conda install -c pytorch pytorch cpuonly -y

```


## Usage

First enter the directory `src` as the CWD.

### Prepare the single-cell AS matrix

```bash
python get_filter/get_filter.py [-h] [--input YOUR_TAB_FILE] [--filter-output YOUR_AS_FILE]

optional arguments:
  -h, --help            Show the most complete help message and exit.
```
| Argument | Type | Default | Description |
|---|---|---|---|
| `--splice_file` | `str` | `../*SJ.out.tab` | Glob path pattern for tab files. The SJ.out.tab file can be obtained by aligning the raw sequencing data in the regular STAR pipeline. |
| `--filter-output` | `str` | `../demo` | Filter result output prefix (will generate *_start.csv and *_end.csv). |
| `--sites` | `int` | `10` | Quality control of the AS matrix, minimum number of expressing cells per AS group. |
| `--samples` | `int` | `1000` | Quality control of the AS matrix, minimum number of expressing sites (sum of AS group) per cell. |


### Data preprocessing

```bash
python preprocess.py [-h] [--splice_file YOUR_AS_FILE] [--gene_file YOUR_RBP_EXPRESSION_FILE] [--output_dir YOUR_OUTPUT_DIR] [--low_threshold YOUR_CLASSIFICATION_THRESHOLD_LOW] [--high_threshold YOUR_CLASSIFICATION_THRESHOLD_HIGH]

optional arguments:
  -h, --help            Show the most complete help message and exit.
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--splice_file` | `str` | `../demo_filter` | Path **prefix** for single-cell AS matrix file (`output files` of the step `Prepare the single-cell AS matrix`). If the path prefix is xxx, we expect there would be xxx_start.csv and xxx_end.csv to represent the 3’ and 5’ AS data. xxx_start.csv and xxx_end.csv can also be obtained from the BAM files by SCASL. |
| `--gene_file` | `str` | `../demo_RBP_expression.csv` | Path to the RBP gene expression CSV file (rows = genes, columns = cells). |
| `--output_dir` | `str` | `../post_process_data/demo/` | Root output directory. Created automatically if it does not exist. |
| `--low_threshold` | `float` | `0.4` | Lower boundary for splice probability categorization. Probabilities **strictly below** this value are assigned **category 1** (low). Must satisfy `0 ≤ low_threshold < high_threshold ≤ 1`. |
| `--high_threshold` | `float` | `0.6` | Upper boundary for splice probability categorization. Probabilities **greater than or equal to** this value are assigned **category 3** (high). Values in `[low_threshold, high_threshold)` are assigned **category 2** (medium). Must satisfy `0 ≤ low_threshold < high_threshold ≤ 1`. |

#### Output Structure

```
<output_dir>/
├── splice_df.csv            # Full categorized splice feature matrix (cells × splice sites)
├── gene_exp_df.csv          # Full RBP gene expression matrix (cells × genes)
├── train_splice_df.csv      # Training set splice features  (80% split)
├── test_splice_df.csv       # Test set splice features      (20% split)
├── train_gene_exp_df.csv    # Training set gene expression
├── test_gene_exp_df.csv     # Test set gene expression
├── k1/
│   ├── train_splice_df.csv
│   ├── test_splice_df.csv
│   ├── train_gene_exp_df.csv
│   ├── test_gene_exp_df.csv
│   ├── splice_df.csv        # Full matrix (copy, for convenience)
│   └── gene_exp_df.csv
├── k2/ ... k5/              # Same structure for folds 2–5
```

### Train the models and explain the models

```bash
python train.py [-h] [-k K] [-i INPUT] [-b] [-o OUTPUT]

optional arguments:

  -h, --help             Show the most complete help message and exit.
```
| Argument | Type | Default | Description |
|---|---|---|---|
| `-k` | `str` | `-1` | column (site) index to train the model with. `-1` means all columns. You can either specify a single column like 0, 1, or 2, or a range like `0-2`, or a list wrapper by quotes and separated by commas like `8, 10, 12`. |
| `-i` | `str` | `../post_process_data/demo` | input directory path (also the `output directory` of the step `Data preprocessing`).|
| `-o` | `str` | `../output/demo` | Output the root directory (subdirectories `k1-k5` will be created and written to the result). |
| `-f` | `str` | `k1,k2,k3,k4,k5` | A comma-separated list of subdirectories to be processed.|

It is recommended to run multiple processes at once. You can either invoke multiple processes manually by specifying different `k` ranges (like `-k "0-399"`, `-k "400-799"`, etc.) or by using slurm and invoking multiple batch jobs.

#### Output Structure
``
base_dir/
├── k1/
│   ├── metric_logs/
│   │   └── <model>/
│   │       └── *.csv              # Must contain: accuracy, site_index, site
│   └── shap_values/
│       └── <model>/
│           └── <column_N>/
│               └── shap_analysis.csv   # First 3 columns: gene, shap1, shap2
├── k2/
│   └── ...
└── ...
```


### Screening of AS regulatory factors

```bash
python reg_summary.py [-h] [-b BASE_DIR]

optional arguments:
  -h, --help            Show the most complete help message and exit.
```
| Argument | Type | Default | Description |
|---|---|---|---|
| `-b` | `str` | `../output/demo` |  Input root directory (contains k1-k5 subdirectories, the `output directory` of the step `Train the models and explain the models`), also the output directory. |
| `-f` | `str` | `k1,k2,k3,k4,k5` | A comma-separated list of subdirectories to be processed.|
| `--threshold-mode` | `str` | `adaptive-knee` | OFeature importance filtering strategy. See details below. |
| `--cumulative-pct` | `float` | `0.8` | Target cumulative contribution ratio, in range (0, 1]. For example, `0.80` retains genes accounting for the top 80% of total importance per site. Only applied when `--threshold-mode adaptive-cumulative` is set.|
| `--shap-threshold` | `float` | `20.0` | Manual absolute SHAP difference threshold. Only applied when `--threshold-mode fixed` is set. |
| `--min-common-folders` | `int` | `5` | Minimum number of folds in which a gene–site pair must appear to be retained. Increasing this value improves result reproducibility. |
| `--contribution-filter` | `str` | `knee` | Global post-hoc filter applied to the final aggregated `Contribution` values. |
| `--top-per-site` | flag | disabled | If set, retains only the top-N RBPs per `AS_Site` × `Direction` group. |
| `--top-n` | `int` | `20` | Number of top RBPs to keep per group when `--top-per-site` is enabled. |
| `--processes` | `int` | `min(4, cpu_count)` | Number of parallel worker processes for fold-level I/O. |

**Mode descriptions:**

- **`adaptive-knee` (default)** — For each AS site, automatically detects the elbow/knee point of the cumulative `|shap1 − shap2|` contribution curve. Retains all features up to and including the knee. No manual threshold required.
- **`adaptive-cumulative`** — For each AS site, retains features until their cumulative contribution reaches the target percentage defined by `--cumulative-pct`.
- **`fixed`** — Applies a global manual threshold; retains only entries where `|shap1 − shap2| > --shap-threshold`.



## Outputs

### Regulator Contribution Summary

The primary output of CASREL is a CSV file named `Regulator_contribution_summary.csv`, containing the predicted RNA splicing regulatory circuitry. This file will be generated in the user-specified output directory.

#### File Structure

| Column | Description |
|--------|------------|
| RBP | RNA binding protein (regulatory factor) |
| AS_Event | Alternative splicing event (splicing site) regulated by the RBP |
| Contribution | Predicted regulatory contribution score |
| Regulation | Regulatory direction (`+` for promotion, `-` for inhibition) |

This output file provides a comprehensive overview of the predicted regulatory relationships between RBPs and their target splicing events, including both the strength and direction of regulation.
