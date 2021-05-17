# imports
import scipy.io
import time
from plotclusters3D import *
from meanshift import *
from meanshift_opt import *
import argparse
import warnings
warnings.simplefilter("ignore")

# SCRIPT USAGE EXAMPLE:
### non-optimized version:
# python Code/debug_img.py --data Data/pts.mat --r 2 --c 4 --opt False

### optimized version:
# python Code/debug_img.py --data Data/pts.mat --r 2 --c 4 --opt True

def str2bool(v):
    """
    Function that converts input string to boolean variable
    :param v: input string
    :return: boolean variable False/True
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    """
    Function that gets parsed input parameters

    :return: parsed parameters
    """
    # parse parameters
    parser = argparse.ArgumentParser(description='Debug algorithm')
    parser.add_argument("--data", type=str, default="", help="Path to the data")
    parser.add_argument("--r", type=int, default=2, help="Search window radius")
    parser.add_argument("--c", type=int, default=4, help="Some  constant  value (c = 4 by default)")
    parser.add_argument("--feature_type", type=int, default=3,
                        help="The dimension of the feature vector as 3D or as 5D (specifying the color and x, y coordinates for each pixel)")
    parser.add_argument("--opt", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Flag to indicate whether we use optimized version of meanshift algorithm or not")

    return parser


def debug(params):
    """
    Function to debug the data set provided with the assignment zip folder

    :param data: n-dimensional dataset consisting of p points
    :param r: the search window radius
    :param c: some  constant  value (c = 4 by default)
    :param opt: a flag to indicate whether we use optimized version of meanshift algorithm or not
    :return: 1) time that the algorithm took to get executed
             2) the class (peak) distribution of points
             3) 3D plot which should give two clusters
    """
    # read the debug data
    data = scipy.io.loadmat(params.data)['data']
    # display the shape
    print(f'data shape: {data.shape}')

    # run the meanshift algorithm
    start_time = time.time()
    if params.opt:
        print(f"Optimization is {params.opt}")
        labels, peaks = meanshift_opt(data, r=params.r, c=params.c)
    else:
        print(f"Optimization is {params.opt}")
        labels, peaks = meanshift(data, r=params.r)
    duration = time.time() - start_time

    print(f'The algorithm took {duration:.2f} s')
    print('\n')

    # check the distribution of points
    y = np.bincount(labels.astype(int))
    ii = np.nonzero(y)[0]

    print("The class (peak) distribution of points:")
    print(np.vstack((ii, y[ii])).T)

    # visualize the points distribution
    plotclusters3D(np.einsum("ij -> ji", data), labels, peaks)


if __name__ == '__main__':
    # parse an input
    parser = get_parser()
    params, unknown = parser.parse_known_args()

    # run the debug function
    debug(params)

