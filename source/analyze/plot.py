import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def get_algs_rspaths_filtered(algs_rspaths, features_chosen):
    algs_rspaths_filtered = {}
    for algcombo, rspathslist in algs_rspaths.items():
        keep_algcombo = True
        features = algcombo.split('-')
        alg, mode, hz, window = features[0], features[1], features[2], None
        if alg not in features_chosen['algs']:
            keep_algcombo = False
        if mode not in features_chosen['modes']:
            keep_algcombo = False
        if hz not in features_chosen['hzs']:
            keep_algcombo = False
        if len(features) == 4:
            window = features[3]
            if window not in features_chosen['windows']:
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
            try:
                rss += list(pd.read_csv(rspath)['RankScores'].values)
            except:
                print(f"\n  FILE NOT FOUND\n    --> {rspath}")
        if len(rss) > 0:
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
    dir_rankscores = "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores"
    algs_rspaths = {
        'DTW-batch-1hz': [
            f"{dir_rankscores}/dtw/testing=batch/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'DTW-online-1hz-1window': [
            f"{dir_rankscores}/dtw/testing=online/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
                        ],
        'DTW-online-1hz-10window': [
            f"{dir_rankscores}/edr/testing=online/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=10/rankscores.csv",
                        ],
        'DTW-batch-3hz': [
            f"{dir_rankscores}/dtw/testing=batch/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv"
                        ],
        'DTW-online-3hz-1window': [
            f"{dir_rankscores}/dtw/testing=online/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
                        ],
        'DTW-online-3hz-10window': [
            f"{dir_rankscores}/dtw/testing=online/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=10/rankscores.csv",
                        ],
        'DTW-batch-5hz': [
            f"{dir_rankscores}/dtw/testing=batch/HZ=5;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'DTW-batch-10hz': [
            f"{dir_rankscores}/dtw/testing=batch/HZ=10;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'DTW-batch-20hz': [
            f"{dir_rankscores}/dtw/testing=batch/HZ=20;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'DTW-batch-50hz': [
            f"{dir_rankscores}/dtw/testing=batch/HZ=50;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'DTW-batch-100hz': [
            f"{dir_rankscores}/dtw/testing=batch/HZ=100;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-batch-1hz': [
            f"{dir_rankscores}/edr/testing=batch/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-online-1hz-1window': [
            f"{dir_rankscores}/edr/testing=online/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
                        ],
        'EDR-online-1hz-10window': [
            f"{dir_rankscores}/edr/testing=online/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=10/rankscores.csv",
                        ],
        'EDR-batch-3hz': [
            f"{dir_rankscores}/edr/testing=batch/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-online-3hz-1window': [
            f"{dir_rankscores}/edr/testing=online/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
                        ],
        'EDR-online-3hz-10window': [
            f"{dir_rankscores}/edr/testing=online/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=10/rankscores.csv",
                        ],
        'EDR-batch-5hz': [
            f"{dir_rankscores}/edr/testing=batch/HZ=5;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-batch-10hz': [
            f"{dir_rankscores}/edr/testing=batch/HZ=10;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-batch-20hz': [
            f"{dir_rankscores}/edr/testing=batch/HZ=20;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-batch-50hz': [
            f"{dir_rankscores}/edr/testing=batch/HZ=50;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'EDR-batch-100hz': [
            f"{dir_rankscores}/edr/testing=batch/HZ=100;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'HTM-online-3hz': [
            f"{dir_rankscores}/htm/testing=online/HZ=3;TESTS=[1, 2, 3];FEATURES=['ys', 'zs', 'dists']/rankscores.csv",  #'xs',
                        ],
        'HTM-online-5hz': [
            f"{dir_rankscores}/htm/testing=online/HZ=5;TESTS=[1, 2, 3];FEATURES=['ys', 'zs', 'dists']/rankscores.csv", #'xs',
                        ],
        'HTM-online-10hz': [
            f"{dir_rankscores}/htm/testing=online/HZ=10;TESTS=[1, 2, 3];FEATURES=['ys', 'zs', 'dists']/rankscores.csv",  #'xs',
                        ],
        'HTM-online-100hz': [
            f"{dir_rankscores}/htm/testing=online/HZ=100;TESTS=[1, 2, 3];FEATURES=['ys', 'zs', 'dists']/rankscores.csv",  #'xs',
                        ],
        'LSTM-batch-1hz': [
            f"{dir_rankscores}/lstm/testing=batch/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-online-1hz-1window': [
            f"{dir_rankscores}/dtw/testing=online/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
                        ],
        'LSTM-online-1hz-10window': [
            f"{dir_rankscores}/dtw/testing=online/HZ=1;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=10/rankscores.csv",
                        ],
        'LSTM-batch-3hz': [
            f"{dir_rankscores}/lstm/testing=batch/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-online-3hz-1window': [
            f"{dir_rankscores}/lstm/testing=online/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
                        ],
        'LSTM-online-3hz-10window': [
            f"{dir_rankscores}/lstm/testing=online/HZ=3;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=10/rankscores.csv",
                        ],
        'LSTM-batch-5hz': [
            f"{dir_rankscores}/lstm/testing=batch/HZ=5;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-online-5hz-1window': [
            f"{dir_rankscores}/lstm/testing=online/HZ=5;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
                        ],
        'LSTM-online-5hz-10window': [
            f"{dir_rankscores}/lstm/testing=online/HZ=5;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=10/rankscores.csv",
                        ],
        'LSTM-batch-10hz': [
            f"{dir_rankscores}/lstm/testing=batch/HZ=10;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-online-10hz-1window': [
            f"{dir_rankscores}/lstm/testing=online/HZ=10;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
                        ],
        'LSTM-online-10hz-10window': [
            f"{dir_rankscores}/lstm/testing=online/HZ=10;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=10/rankscores.csv",
                        ],
        'LSTM-online-100hz-1window': [
            f"{dir_rankscores}/lstm/testing=online/HZ=100;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=1/rankscores.csv",
                        ],
        'LSTM-online-100hz-10window': [
            f"{dir_rankscores}/lstm/testing=online/HZ=100;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/window=10/rankscores.csv",
                        ],
        'LSTM-batch-20hz': [
            f"{dir_rankscores}/lstm/testing=batch/HZ=20;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-batch-50hz': [
            f"{dir_rankscores}/lstm/testing=batch/HZ=50;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
        'LSTM-batch-100hz': [
            f"{dir_rankscores}/lstm/testing=batch/HZ=100;TESTS=[1, 2, 3];FEATURES=['xs', 'ys', 'zs', 'dists']/rankscores.csv",
                        ],
    }


    features_chosen = {
        'algs': ['LSTM', 'HTM', 'DTW', 'EDR'],  #, 'HTM'
        'hzs': ['100hz', '50hz', '20hz', '10hz', '5hz', '3hz'],  #, '100hz', '1hz'
        'modes': ['online'],  #'batch', 'online'
        'windows': ['1window']  #, '1window',
    }

    dir_out = "/Users/samheiserman/Desktop/PhD/Motion-Print/output/rank_scores"
    out_label = ''
    for feat, chosen in features_chosen.items():
        out = f"{feat}={','.join(chosen)}--"
        out_label += out

    out_title = "RankScores"
    algs_rspaths_filtered = get_algs_rspaths_filtered(algs_rspaths, features_chosen)
    print("algs_rspaths_filtered...")
    for algcombo, rspathslist in algs_rspaths_filtered.items():
        print(f"  {algcombo}")
        for rspath in rspathslist:
            print(f"    --> {rspath}")


    boxplot_rankscores(algs_rspaths_filtered, dir_out, label=out_label, title=out_title)
