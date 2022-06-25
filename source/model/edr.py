import editdistance


def get_edr_dist(data1, data2):
    dist = 0
    # Get dist for each column (since 1D method)
    for col_i in range(data1.shape[1]):
        data1_ = data1[:, col_i]
        data2_ = data2[:, col_i]
        dist += editdistance.eval(data1_, data2_)
    return dist