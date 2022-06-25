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
    hz = 10
    tests = [1, 2, 3]
    feats = ['xs', 'ys', 'zs', 'dists']
    algs_rspaths = {
        'lstm-batch': f"/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/training=batch/HZ={hz};TESTS={tests};FEATURES={feats}/rankscores.csv",
        'lstm-online-window=1': f"/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/training=online/window=1/HZ={hz};TESTS={tests};FEATURES={feats}/rankscores.csv",
        'lstm-online-window=10': f"/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/training=online/window=10/HZ={hz};TESTS={tests};FEATURES={feats}/rankscores.csv",
        'lstm-online-window=100': f"/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/training=online/window=100/HZ={hz};TESTS={tests};FEATURES={feats}/rankscores.csv",
        'lstm-online-window=1000': f"/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/training=online/window=1000/HZ={hz};TESTS={tests};FEATURES={feats}/rankscores.csv",
    }
    dir_out = "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores"
    boxplot_rankscores(algs_rspaths, dir_out)
