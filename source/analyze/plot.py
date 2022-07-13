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
        if len(rspaths) == 0:
            print("  files found --> NONE")
            continue
        print(f"  files found --> {len(rspaths)}")
        plotbars_rss[plotbar] = []
        for rspath in rspaths:
            rs = list(pd.read_csv(rspath)['RankScores'].values)
            plotbars_rss[plotbar] += rs
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
    algs, testmodes, hzs, feats = [], [], [], []
    for plotbar, filtertraits in plotbars_filtertraits.items():
        algs.append(filtertraits['ALG='].upper())
        hzs.append(filtertraits['HZ='])
        feats.append(filtertraits['FEATURES='])
        if 'TESTMODE=' in filtertraits:
            testmodes.append(filtertraits['TESTMODE='])
    algs, testmodes, hzs, feats = set(algs), set(testmodes), set(hzs), set(feats)
    return sorted(algs), sorted(testmodes), sorted(hzs), sorted(feats)


if __name__ == '__main__':

    config = load_config(get_args().config_path)

    algs, testmodes, hzs, feats = get_traits(config['plotbars_filtertraits'])
    all_rs_paths = get_dirfiles(config['dir_rankscores'], files_types=['csv'])
    plotbars_rss = get_plotbars_rss(all_rs_paths, config['plotbars_filtertraits'])

    title = f"algs={','.join(algs)}; testmodes={','.join(testmodes)}; hzs={','.join(hzs)}"  #; feats={','.join(feats)}
    outpath = os.path.join(config['dir_out'], f"{title}.png")

    plot_rs_boxplot(plotbars_rss, title, outpath, config['xlabel'], config['ylabel'])
