import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)
from source.utils.utils import get_dirfiles, load_config, get_args


def get_plotbars_rss(all_rs_paths, plotbars_filtertraits):
    plotbars_rss = {}
    for plotbar, filtertraits in plotbars_filtertraits.items():
        print(f"\n{plotbar}")
        # gather eligible paths
        rspaths = []
        for rspath in all_rs_paths:
            keep = True
            for trait, val in filtertraits.items():
                if trait + val not in rspath:
                    keep = False
            if keep:
                rspaths.append(rspath)
        # gather rankscores from eligibles
        print(f"  files found --> {len(rspaths)}")
        if len(rspaths) == 0:
            continue
        plotbars_rss[plotbar] = []
        for rspath in rspaths:
            rs = list(pd.read_csv(rspath)['RankScores'].values)
            plotbars_rss[plotbar] += rs

    print(f"\nplotbars_rss...")
    for k,v in plotbars_rss.items():
        print(f"  {k} = {v}")

    return plotbars_rss


def histplot_rankscores(algs_rspaths, dir_out):
    outpath = os.path.join(dir_out, 'algs_rankscores_hist.png')
    plt.cla()
    for alg, rspath in algs_rspaths.items():
        rs = pd.read_csv(rspath)['RankScores'].values
        plt.hist(rs, label=alg, alpha=0.5)
    plt.legend()
    plt.xlabel('Rank Score')
    plt.ylabel('Frequency')
    plt.savefig(outpath)


def plot_rs_boxplot(plotbars_rss, title, outpath, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.boxplot(plotbars_rss.values())
    ax.set_xticklabels(plotbars_rss.keys(), rotation=90)
    ax.yaxis.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(0.5, label="Random Avg Score", color='red', linestyle='--', linewidth=1)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.savefig(outpath, bbox_inches="tight")


def get_traits(plotbars_filtertraits):
    traits_vals = {}
    for plotbar, filtertraits in plotbars_filtertraits.items():
        for trait, trait_val in filtertraits.items():
            try:
                traits_vals[trait].append(trait_val)
            except:
                traits_vals[trait] = [trait_val]
    traits_vals = {t:list(set(v)) for t,v in traits_vals.items()}
    return traits_vals


if __name__ == '__main__':

    config = load_config(get_args().config_path)

    all_rs_paths = get_dirfiles(config['dir_rankscores'], files_types=['csv'])
    all_rs_paths = [f for f in all_rs_paths if 'rankscores.csv' in f]
    plotbars_rss = get_plotbars_rss(all_rs_paths, config['plotbars_filtertraits'])

    traits_vals = get_traits(config['plotbars_filtertraits'])
    title = ""
    for trait, vals in traits_vals.items():
        title += f"{trait}={vals}; "

    outpath = os.path.join(config['dir_out'], f"{title}.png")
    plot_rs_boxplot(plotbars_rss, title, outpath, config['xlabel'], config['ylabel'])
