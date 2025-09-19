import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


plt.switch_backend('agg')


def split_start_end(name):
    splits = name.split('_')
    start = '_'.join(splits[:2])
    end = '_'.join([splits[0], splits[2]])
    return start, end


def make_start_end_df(df):
    sites = df['Site']
    starts = []
    ends = []
    for s in sites:
        start, end = split_start_end(s)
        starts.append(start)
        ends.append(end)
    df['start'] = starts
    df['end'] = ends
    return df


def repeat_filter_end(df, make_start_end=True, save=False):
    if make_start_end:
        df = make_start_end_df(df)
    repeat_df_end = df[df.duplicated('end', keep=False)]
    if save:
        repeat_df_end.to_csv('process_result/repeat_df_end.csv')
    return repeat_df_end


def repeat_filter_start(df, make_start_end=True, save=False):
    if make_start_end:
        df = make_start_end_df(df)
    repeat_df_start = df[df.duplicated('start', keep=False)] 
    if save:
        repeat_df_start.to_csv('process_result/repeat_df_start.csv')
    return repeat_df_start


###########################

def draw_hist(count, split='site', log=True):
    sns.set()
    plt.figure(figsize=(10, 6))
    sns.histplot(np.log(count) if log else count, bins=30, kde=True)
    if not os.path.exists('img'):
        os.makedirs('img')
    plt.xlabel('log value of the number of non-NaN data items' if log else 'number of non-NaN data itemsao')
    plt.savefig('img/%s_hist.png' % (split, ))
    print('the %s histogram is saved at ./img/%s_hist.png' % (split, split))
    print('the descriptions of the non-NaN data of sites are shown below')
    print(count.describe(percentiles=np.arange(0, 100, 10) / 100.))
    plt.clf()


def qc_filter_sites(df_repeat, thres=10, draw_hist_only=False, log=True):
    site_count = df_repeat.count(axis=1)
    draw_hist(site_count, 'site', log)
    if not draw_hist_only:
        df_site = df_repeat.copy()
        df_site['site_count'] = site_count.values   
        df_site_filtered = df_site[df_site['site_count'] > thres]
        return df_site_filtered    
    

def qc_filter_samples(df_site_filtered, thres=1000, draw_hist_only=False, log=True):
    sample_count = df_site_filtered.count(axis=0)
    draw_hist(sample_count, 'sample', log)
    if not draw_hist_only:
        df_sample = df_site_filtered.copy()
        df_sample.loc['sample_count'] = sample_count.values
        df_sample_filtered = df_sample.loc[:, df_sample.loc['sample_count'] > thres]
        return df_sample_filtered


def thres_filter(df, samples_ps, sites_ps, by='start'):
    groups = df.groupby(by=by)
    result = []
    for s in tqdm(groups):
        r = s[1].copy().iloc[:, 1:-2]
        sums = r.sum(axis=0)
        counts = r.count(axis=1)
        r[counts < samples_ps] = np.nan
        r.loc[:, sums < sites_ps] = np.nan
        result.append(r)
    result = pd.concat(result)
    result.insert(0, 'Site', df['Site'])
    result[['start', 'end']] = df[['start', 'end']]
    return result


def filter(junc_mat, samples_ps, sites_ps, sites_thres, samples_thres, log=True):
    print('reading file...')
    df = pd.read_csv(junc_mat)
    print('done.')
    df = df.rename(columns={'Unnamed: 0': 'Site'})
    # df = df.set_index('Site')
    print('executing repeat filter...')
    rfs = repeat_filter_start(df)
    rfe = repeat_filter_end(df)

    # 我改了这里
    tfs = thres_filter(rfs, samples_ps, sites_ps, by='start')
    tfe = thres_filter(rfe, samples_ps, sites_ps, by='end')

    print('executing sites filter...')
    dfs = qc_filter_sites(tfs, sites_thres, log=log)
    dfe = qc_filter_sites(tfe, sites_thres, log=log)
#   print(dfs.sum(axis=1).isnull())
#   print(dfe.sum(axis=1).isnull())

    print('done.')

    print('executing final repeat filter...')
    dfs = dfs.iloc[:-1, :-3]
    dfe = dfe.iloc[:-1, :-3]

    dfs = repeat_filter_start(dfs)
    dfe = repeat_filter_end(dfe)
    print('done.')

    df_repeat = pd.concat([dfs, dfe])

    if log:
        sites_thres = np.exp(sites_thres)
        samples_thres = np.exp(samples_thres)

    print('executing sample filter...')
    df_sample_filtered = qc_filter_samples(df_repeat, samples_thres, log=log)
    print('done.')

    dfs = dfs[df_sample_filtered.columns]
    dfe = dfe[df_sample_filtered.columns]
    
    dfs = repeat_filter_start(dfs, save=True).sort_values(by='start').set_index('Site')
    dfe = repeat_filter_end(dfe, save=True).sort_values(by='end').set_index('Site')
    
    df_repeat = pd.concat([dfs, dfe])
#    print(df_repeat.count())

    return dfs, dfe


def draw_histogram(junc_mat, log):
    print('reading file...')
    df = pd.read_csv(junc_mat)
    print('done.')
    df = df.rename(columns={'Unnamed: 0': 'Site'})
    df = df.set_index('Site')
    print('executing repeat filter...')
    df_repeat = repeat_filter(df, save=False, combine=True)
    print('done.')
    print('computing histograms for non-NaN values for sites and samples')
    site_count = qc_filter_sites(df_repeat, draw_hist_only=True, log=log)
    samples_count = qc_filter_samples(df_repeat, draw_hist_only=True, log=log)
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--junc', default='process_result/junc_matrix.csv', type=str)
    parser.add_argument('-o', '--output', default='process_result/filtered_matrix', type=str)
    parser.add_argument('--st', dest='sites_ps', default=10, type=int)
    parser.add_argument('--sp', dest='samples_ps', default=5, type=int)
    parser.add_argument('--sites', dest='sites_thres', default=10, type=int)
    parser.add_argument('--samples', dest='samples_thres', default=1000, type=int)
    parser.add_argument('--mode', choices=['filter', 'hist'], type=str, default='filter')
    parser.add_argument('--log', action='store_true', help='whether to perform logorithm on the counts')
    args = parser.parse_args()
    if args.mode == 'filter':
        dfs, dfe = filter(args.junc, args.samples_ps, args.sites_ps, args.sites_thres, args.samples_thres, args.log)
        print('saving...')
        dfs.to_csv(args.output + '_start.csv')
        dfe.to_csv(args.output + '_end.csv')
        print('done.')
    else:
        draw_histogram(args.junc, args.log) # 这里