import editdistance


def get_edr_dist(data1, data2):
    dist = 0
    for c in data1:
        dist += editdistance.eval(data1[c].values, data2[c].values)
    return dist