import os
import sys

import pandas as pd

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import make_dir


def get_rankscore(subjtest, subjsdists):
    # count subjects w/dist > subjtest (the more the better)
    further_than_subjtest = {subj: dist for subj, dist in subjsdists.items() if dist > subjsdists[subjtest]}
    return len(further_than_subjtest) / (len(subjsdists) - 1)


def write_rankscores(subjstest_subjsdists, dir_output):
    make_dir(dir_output)
    subjects_rankscores = {}
    # Get rankscores
    for subjtest, subjsdists in subjstest_subjsdists.items():
        subjects_rankscores[subjtest] = get_rankscore(subjtest, subjsdists)
    # Write out
    rankscores_df = pd.DataFrame(subjects_rankscores, index=[0]).T
    rankscores_df.columns = ['RankScores']
    rankscores_df.to_csv(os.path.join(dir_output, f'rankscores.csv'))
    return subjects_rankscores
