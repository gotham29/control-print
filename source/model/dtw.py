import dtw


def get_dtw_dist(data1, data2):
    return dtw.dtw(data1, data2).distance