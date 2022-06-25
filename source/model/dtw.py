from fastdtw import fastdtw

def get_dtw_dist(data1, data2):
    distance, path = fastdtw(data1, data2) #dist='euclidean'
    return distance