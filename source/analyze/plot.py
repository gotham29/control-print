import os

import matplotlib.pyplot as plt
import pandas as pd


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

def boxplot_rankscores(algs_rspaths, dir_out, desired_order_list = ['dtw', 'edr', 'lstm']):
    outpath = os.path.join(dir_out, 'algs_rankscores_box.png')
    fig, ax = plt.subplots()
    algs_rs = {}
    for alg, rspath in algs_rspaths.items():
        algs_rs[alg] = pd.read_csv(rspath)['RankScores'].values
    algs_rs = {k: algs_rs[k] for k in desired_order_list}
    ax.boxplot(algs_rs.values())
    ax.set_xticklabels(algs_rs.keys())
    plt.savefig(outpath)

""" TEST """
if __name__ == '__main__':
    algs_rspaths = {
        'lstm': "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/HZ=5;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
        'dtw': "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/dtw/HZ=5;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
        'edr': "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/edr/HZ=5;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
    }
    dir_out = "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores"
    # histplot_rankscores(algs_rspaths, dir_out)
    boxplot_rankscores(algs_rspaths, dir_out)
