# imports
import numpy as np
from scipy.spatial.distance import cdist, pdist


# function
def find_peak_opt(data, idx, r, c=4, threshold=0.01):
    """
    Optimized function that performs the peak searching processes for each point
    computing its associated peak by first defining a spherical window at the data point of radius r and computing
    the mean of the points that lie within the window. The algorithm then shifts the window to the mean and repeats
    until convergence, i.e.  the  shift  is  under  some  threshold t (for  example, t =  0.01).
    With  each iteration the window will shift to a more densely populated portion of the data set
    until a peak is reached, where the  data  is  equally  distributed  in  the  window.

    :param data: n-dimensional dataset consisting of p points
    :param idx: the column index of the data point for which we wish to compute its associated density peak
    :param r: the search window radius
    :param threshold: the  shift  is  under  some  threshold
    :param c: some  constant  value (c = 4 by default)
    :return: 1) associated peak with a data point
             2) a vector storing a 1 for each point that is a distance of r/c from the path and 0 otherwise.
    """
    cpts = np.zeros([1, data.shape[1]])
    data_point = data[:, idx]
    diff = np.ones(data_point.shape)

    while all(x > threshold for x in diff):
        distances = cdist(data.T, data_point.reshape((-1, 1)).T, metric='euclidean').reshape(-1)
        # second speedup
        cpts[:, distances <= r/c] = 1
        found_points = data[:, distances < r]
        peak = np.mean(found_points, axis=1)
        diff = pdist(np.array([data_point, peak]))
        data_point = peak

    return np.array(peak), cpts
