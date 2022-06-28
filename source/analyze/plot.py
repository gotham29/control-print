import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def get_algs_rspaths_filtered(algs_rspaths, substrs_req):
    algs_rspaths_filtered = {}
    for algcombo, rspathslist in algs_rspaths.items():
        keep_algcombo = True
        for substr in substrs_req:
            if substr not in algcombo:
                keep_algcombo = False
        if keep_algcombo:
            algs_rspaths_filtered[algcombo] = rspathslist
    return algs_rspaths_filtered


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


def boxplot_rankscores(algs_rspaths, dir_out, label=None, title=None):
    outname = 'algs_rankscores_box.png'
    if label is not None:
        outname = outname.replace('.png', f"--{label}.png")

    outpath = os.path.join(dir_out, outname)
    fig, ax = plt.subplots()
    algs_rs = {}
    for alg, rspathlist in algs_rspaths.items():
        rss = []
        for rspath in rspathlist:
            rss += list(pd.read_csv(rspath)['RankScores'].values)
        algs_rs[alg] = rss
    ax.boxplot(algs_rs.values())
    ax.set_xticklabels(algs_rs.keys(), rotation=90)
    ax.yaxis.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("Experimental Settings")
    plt.ylabel("Rank Scores")
    plt.axhline(0.5, label="Random Avg Score", color='red', linestyle='--', linewidth=1)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.savefig(outpath, bbox_inches="tight")


""" TEST """
if __name__ == '__main__':
    algs_rspaths = {
        'DTW-batch-1hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/dtw/testing=batch/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'DTW-online-1hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/dtw/testing=online/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
        ],
        'DTW-batch-3hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/dtw/testing=batch/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv"
                        ],
        'DTW-batch-5hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/dtw/testing=batch/HZ=5;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'DTW-batch-10hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/dtw/testing=batch/HZ=10;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'DTW-batch-20hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/dtw/testing=batch/HZ=20;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'DTW-batch-50hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/dtw/testing=batch/HZ=50;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'DTW-batch-100hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/dtw/testing=batch/HZ=100;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-batch-1hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/edr/testing=batch/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-online-1hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/edr/testing=online/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
        ],
        'EDR-batch-3hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/edr/testing=batch/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-batch-5hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/edr/testing=batch/HZ=5;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-batch-10hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/edr/testing=batch/HZ=10;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-batch-20hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/edr/testing=batch/HZ=20;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-batch-50hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/edr/testing=batch/HZ=50;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-batch-100hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/edr/testing=batch/HZ=100;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-batch-1hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/testing=batch/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-online-1hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/dtw/testing=online/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
        ],
        'LSTM-batch-3hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/testing=batch/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-batch-5hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/testing=batch/HZ=5;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-batch-10hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/testing=batch/HZ=10;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-batch-20hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/testing=batch/HZ=20;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-batch-50hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/testing=batch/HZ=50;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-batch-100hz': [
            "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores/lstm/testing=batch/HZ=100;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
    }

    substrs_req = [
        '1hz',
        'LSTM',
        # 'online',
                   ]
    dir_out = "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores"
    out_label = '--'.join(substrs_req) #"batch--dtw,edr,lstm--1,3,5,10,20,50,100hz"
    out_title = "RankScores -- Mode=Batch; Tests=1,2,3"
    algs_rspaths_filtered = get_algs_rspaths_filtered(algs_rspaths, substrs_req)
    print("algs_rspaths_filtered...")
    for algcombo, rspathslist in algs_rspaths_filtered.items():
        print(f"  {algcombo}")
        for rspath in rspathslist:
            print(f"    --> {rspath}")


    boxplot_rankscores(algs_rspaths_filtered, dir_out, label=out_label, title=out_title)
