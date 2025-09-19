import argparse
import os
from typing import Tuple


from better_process_star import run_better_process

from fliter_final import filter as run_filter_core


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run_pipeline(
    input_glob: str,
    max_na_ratio: float,
    process_output_csv: str,
    filter_output_prefix: str,
    samples_ps: int,
    sites_ps: int,
    sites_thres: int,
    samples_thres: int,
    log: bool
) -> Tuple[str, str, str]:

    print('Step 1/2: Run data preprocessing ...')
    ensure_dir(os.path.dirname(os.path.abspath(process_output_csv)))
    process_csv = run_better_process(input_glob, max_na_ratio, process_output_csv)

    # 2) 过滤：调用 fliter_final.filter
    print('Step 2/2: Run AS event selection and filtering ...')
    # fliter_final.filter will try to write to process_result/repeat_df_*.csv when saving

    ensure_dir('process_result')

    dfs, dfe = run_filter_core(
        process_csv,
        samples_ps,
        sites_ps,
        sites_thres,
        samples_thres,
        log
    )

    ensure_dir(os.path.dirname(os.path.abspath(filter_output_prefix)))
    start_out = f'{filter_output_prefix}_start.csv'
    end_out = f'{filter_output_prefix}_end.csv'
    dfs.to_csv(start_out)
    dfe.to_csv(end_out)
    print('Filter results saved:')
    print('  -', start_out)
    print('  -', end_out)
    return process_csv, start_out, end_out


def build_cli():
    p = argparse.ArgumentParser(
        description='Get the tab file and filter file from STAR mapping.'
    )

    p.add_argument('-i', '--input', required=True, help='Glob path pattern for tab files (e.g. /path/to/*.tab).')
    p.add_argument('-r', '--max-na-ratio', type=float, default=0.999, help='Maximum NA ratio threshold (default 0.999).')
    p.add_argument('-p', '--process-output', required=True, help='Output path of single-cell junction reads matrix.')

    p.add_argument('--st', dest='sites_ps', default=10, type=int, help='quality control of the initial junction matrix, minimum number of expressing cells per site.')
    p.add_argument('--sp', dest='samples_ps', default=5, type=int, help='quality control of the initial junction matrix, minimum number of expressing sites per cell.')
    p.add_argument('--sites', dest='sites_thres', default=10, type=int, help='quality control of the AS matrix, minimum number of expressing cells per AS group.')
    p.add_argument('--samples', dest='samples_thres', default=1000, type=int, help='quality control of the AS matrix, minimum number of expressing sites (sum of AS group) per cell.')
    p.add_argument('--log', action='store_true', help='whether to output the quality distribution histogram to help determine the quality control standard, logical parameter (T or F).')


    p.add_argument('-o', '--filter-output', default='process_result/filtered_matrix',
                   help='Filter result output prefix (will generate *_start.csv and *_end.csv).')

    return p


def main():
    args = build_cli().parse_args()
    run_pipeline(
        input_glob=args.input,
        max_na_ratio=args.max_na_ratio,
        process_output_csv=args.process_output,
        filter_output_prefix=args.filter_output,
        samples_ps=args.samples_ps,
        sites_ps=args.sites_ps,
        sites_thres=args.sites_thres,
        samples_thres=args.samples_thres,
        log=args.log
    )


if __name__ == '__main__':
    main()