# imports
from find_peak import *


# function
def meanshift(data, r):
    """
    Function which calls findpeak for each point and then assigns a label to each point according to its peak.
    Peaks are compared after each call to the findpeak function and for similar peaks to be merged.
    Two peaks are considered to be the same if the distance between them is smaller than r/2.
    Also, if the peak of a data point is found to already exist in peaks then for simplicity
    its computed peak is discarded and it is given the label of the associated peak in peaks.

    :param data: n-dimensional dataset consisting of p points
    :param r: the search window radius
    :return: 1) labels - a vector containing the label for each data point
             2) peaks - is a matrix storing the density peaks found using meanshift as its columns
    """
    # initialization
    labels = np.zeros([1, data.shape[1]]) - 1
    peaks = []
    labels_counter = 0
    labels_values = []

    # process all points (pixels)
    for i in range(data.shape[1]):
        # if there is no class assigned to a consider point (pixel) - the label -1 means it has no class
        if labels[:, i] == -1:
            # calculate a peak of the current point
            peak = find_peak(data, i, r)

            # the beginning of the clustering, we have just a one peak
            if len(peaks) == 0:
                peaks.append(peak)
                labels_values.append(labels_counter)
                labels[:, i] = labels_counter

            # check if the peak of a data point is found to already exist in peaks
            else:
                pick_distances = cdist(np.array(peaks), peak.reshape(-1, 1).T)

                # the peak of a data point is found to already exist in peaks
                if pick_distances[np.argmin(pick_distances[:, 0]), 0] < r/2:
                    labels[:, i] = labels_values[np.argmin(pick_distances[:, 0])]

                # we found a new peak
                else:
                    peaks.append(peak)
                    labels_counter += 1
                    labels[:, i] = labels_counter
                    labels_values.append(labels_counter)

        else:
            pass

    return labels[0], np.array(peaks)
