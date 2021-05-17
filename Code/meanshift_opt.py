# imports
from find_peak_opt import *
from collections import deque


# function
def meanshift_opt(data, r, c=4, threshold=0.01):
    """
    Function which calls find_peak_opt for each point and then assigns a label to each point according to its peak.
    Peaks are compared after each call to the find_peak_opt function and for similar peaks to be merged.
    Two peaks are considered to be the same if the distance between them is smaller than r/2.
    Also, if the peak of a data point is found to already exist in peaks then for simplicity
    its computed peak is discarded and it is given the label of the associated peak in peaks.

    :param data: n-dimensional dataset consisting of p points
    :param r: the search window radius
    :param threshold: the  shift  is  under  some  threshold
    :param c: some  constant  value (c = 4 by default)
    :return: 1) labels - a vector containing the label for each data point
             2) peaks - is a matrix storing the density peaks found using meanshift_opt as its columns
    """
    # initialization
    labels = np.full([1, data.shape[1]], -1)
    peaks = deque()
    labels_counter = 0
    labels_values = deque()

    end = False

    while not end:
        avail_indices = np.nonzero(labels == -1)[1]

        ## Speedup: if all labels were assigned to clusters
        if len(avail_indices) == 0:
            end = True

        ## Speedup: we have a point without a label
        else:
            i = avail_indices[0]
            peak, cpts = find_peak_opt(data=data, idx=i, r=r, c=c, threshold=threshold)

            # the beginning of the clustering, we have just a one peak
            if len(peaks) == 0:
                peaks.append(peak)
                labels_values.append(labels_counter)
                labels[:, i] = labels_counter

                ## First speedup
                dist = cdist(data.T, peak.reshape((-1, 1)).T, metric='euclidean')
                labels[:, np.where(np.any(dist < r, axis=1))] = labels[:, i]

            # calculate a peak of the current point and check
            # if the peak of a data point is found to already exist in peaks
            else:
                pick_distances = cdist(np.array(peaks), peak.reshape(-1, 1).T, metric='euclidean')

                # a similar peak was found
                if pick_distances[np.argmin(pick_distances[:, 0]), 0] < r / 2:
                    labels[:, i] = labels_values[np.argmin(pick_distances[:, 0])]

                # the peak of a data point is not found to already exist in peaks
                else:
                    # new peak
                    peaks.append(peak)
                    labels_counter += 1
                    labels[:, i] = labels_counter
                    labels_values.append(labels_counter)

                    ## First speedup
                    dist = cdist(data.T, peak.reshape((-1, 1)).T, metric='euclidean')
                    labels[:, np.where(np.any(dist <= r, axis=1))] = labels[:, i]

            ## Second speedup
            labels[:, np.where(np.any(cpts == 1, axis=0))] = labels[:, i]

    return labels[0], np.array(peaks)
