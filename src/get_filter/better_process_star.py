import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from json import dump
import argparse
import os


def run_better_process(input_glob: str, max_na_ratio: float, output_csv: str):

    files = list(glob(input_glob))
    if len(files) == 0:
        raise FileNotFoundError(f'No files were found in path: {input_glob}')


    samples = [os.path.basename(f).split('.')[0] for f in files]
    with open('./samples.json', 'w') as f:
        dump(samples, f)

    num_samples = len(samples)
    print('Found', num_samples, 'samples')

    print('pick good sites...')

    sites = {}
    for f in tqdm(files):
        df = pd.read_csv(f, sep='\t', header=None)
        for line in df.values:
            site = '_'.join([str(line[0]), str(line[1]), str(line[2])])
            if site in sites:
                sites[site] += 1
            else:
                sites[site] = 0


    good_sites = [s for s, c in sites.items() if c > (1 - max_na_ratio) * num_samples]
    with open('./good_sites.json', 'w') as f:
        dump(good_sites, f)
    num_sites = len(good_sites)
    print('Selected', num_sites, 'good sites.')
    print('good sites saved to json file.')


    good_sites_dict = {s: i for i, s in enumerate(good_sites)}
    unpivot = []
    print('creating unpivoted matrix...')
    for i, f in enumerate(tqdm(files)):
        df = pd.read_csv(f, sep='\t', header=None)
        for line in df.values:
            site = '_'.join([str(line[0]), str(line[1]), str(line[2])])
            if site in good_sites_dict:
                unpivot.append((good_sites_dict[site], i, line[6]))

    unpivot = np.array(unpivot)
    np.save('unpivot.npy', unpivot)
    print('unpivoted matrix saved to npy file.')


    pivot = np.ones((num_sites, num_samples)) * np.nan
    print('pivoting the data...')
    for row, col, val in tqdm(unpivot):
        pivot[row][col] = val

    df = pd.DataFrame(data=pivot, index=good_sites, columns=samples)


    out_dir = os.path.dirname(os.path.abspath(output_csv))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_csv, na_rep='', sep=',')
    print('pivoted matrix saved to csv:', output_csv)
    return output_csv


def build_cli():
    p = argparse.ArgumentParser(description='Extract and filter single-cell junction reads to create a single-cell junction read matrix.')
    p.add_argument('-i', '--input', required=True, help='Glob path pattern for tab files (e.g. /path/to/*.out.tab).')
    p.add_argument('-r', '--max-na-ratio', type=float, default=0.999, help='Maximum NA ratio threshold (default 0.999).')
    p.add_argument('-o', '--output', required=True, help='Output path of single-cell junction reads matrix.')
    return p


if __name__ == '__main__':
    args = build_cli().parse_args()
    run_better_process(args.input, args.max_na_ratio, args.output)