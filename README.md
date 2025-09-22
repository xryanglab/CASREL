# CASREL
Cell-specific alternative splicing regulation inference via explainable learning. 

The `demo` directory contains example notebooks and minimal test datasets to help users get started with CASREL. These materials demonstrate the complete workflow using a simplified dataset while serving as templates for analyzing your own data.


## Getting start ##

### Dependencies and requirements

Latest release of

- Python3
- PyTorch
- NumPy
- Pandas
- Sci-kit Learn
- LightGBM
- EasyDict

You can also refer to the packages and their corresponding versions in `requirements.txt`.

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
  -h, --help            show this help message and exit

  --splice_file YOUR_AS_FILE
                        Glob path pattern for tab files (e.g. /path/to/*SJ.out.tab). The SJ.out.tab file can be obtained by aligning the raw sequencing data in the regular STAR pipeline.

  --filter-output YOUR_AS_FILE
                        Filter result output prefix (will generate *_start.csv and *_end.csv).
```

### Data preprocessing

```bash
python preprocess.py [-h] [--splice_file YOUR_AS_FILE] [--gene_file YOUR_RBP_EXPRESSION_FILE] [--output_dir YOUR_OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit

  --splice_file YOUR_AS_FILE
                        path to the single-cell AS matrix file ('output files' of the step 'Prepare the single-cell AS matrix'). If the path is xxx, we expect there would be xxx_start.csv and xxx_end.csv to represent the 3’ and 5’ AS data. xxx_start.csv and xxx_end.csv can also be obtained from the BAM files by SCASL.

  --gene_file YOUR_RBP_EXPRESSION_FILE
                        path to the single-cell RBP expression file. Should be a csv file.

  --output_dir YOUR_OUTPUT_DIR
                        output directory path.
```

### Train the models and explain the models

```bash
python train.py [-h] [-k K] [-i INPUT] [-b] [-o OUTPUT]

optional arguments:

  -h, --help            show this help message and exit.

  -k K                  column (site) index to train the model with. -1 means all columns. You can either specify a single column like 0, 1, or 2, or a range like 0-2, or a list wrapper by quotes
                        and separated by commas like "8, 10, 12".

  -i INPUT, --input INPUT
                        input directory path (also the 'output directory' of the step 'Data preprocessing').

  -b, --batch           Whether to enable batch mode. If this is enabled, a single model will try to capture and output the predictions for all the sites specified.

  -o OUTPUT, --output OUTPUT
                        output directory path.
```


It is recommended to run multiple processes at once. You can either invoke multiple processes manually by specifying different `k` ranges (like `-k "0-399"`, `-k "400-799"`, etc.) or by using slurm and invoking multiple batch jobs.


### Screening of AS regulatory factors

```bash
python reg_summary.py [-h] [-b BASE_DIR]

optional arguments:
  -h, --help            show this help message and exit.

  -b BASE_DIR, --base BASE_DIR
                        Input root directory (contains k1-k5 subdirectories, the 'output directory' of the step 'Train the models and explain the models'), also the output directory. Defaults to the current directory.
```

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
