# imports
import itertools
import time
from skimage.color import lab2rgb, rgb2lab
from plotclusters3D import *
from meanshift import *
from meanshift_opt import *
import argparse
import matplotlib.image as mpimg
import warnings
warnings.simplefilter("ignore")



# SCRIPT USAGE EXAMPLE:
### non-optimized version:
# python Code/imageSegmentation.py --im Data/img3.jpg --r 2 --c 4 --feature_type 5 --cloud True --opt False --save True

### optimized version:
# python Code/imageSegmentation.py --im Data/img3.jpg --r 2 --c 4 --feature_type 5 --cloud True --opt True --save True

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
    parser.add_argument("--im", type=str, default="", help="Path to the image")
    parser.add_argument("--r", type=int, default=2, help="Search window radius")
    parser.add_argument("--c", type=int, default=4, help="Some  constant  value (c = 4 by default)")
    parser.add_argument("--feature_type", type=int, default=3, help="The dimension of the feature vector as 3D or as 5D (specifying the color and x, y coordinates for each pixel)")
    parser.add_argument("--cloud", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Flag to indicate whether we plot a 3D cloud of points distribution")
    parser.add_argument("--opt", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Flag to indicate whether we use optimized version of meanshift algorithm or not")
    
    parser.add_argument("--save", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Flag to indicate whether we want to save a segmented image")

    return parser

def imageSegmentation(params):
    """
    Function that performs the actual segmentation.

    :param im: Path to the image
    :param r: Search window radius
    :param c: Some  constant  value (c = 4 by default)
    :param feature_type: The dimension of the feature vector as 3D or as 5D (specifying the color and x, y coordinates for each pixel)
    :param opt: Flag to indicate whether we use optimized version of meanshift algorithm or not
    :return:
    """
    img_to_rgb = plt.imread(params.im)
    img_to_lab = rgb2lab(img_to_rgb / 255.)

    if params.feature_type == 3:
        print(f"The feature vector as 3D with the color channels.")
        processed_img = img_to_lab
        processed_img_reshaped = processed_img.reshape((processed_img.shape[0] * processed_img.shape[1], processed_img.shape[2])).T

    elif params.feature_type == 5:
        print(f"The feature vector as 5D with the color channels and x, y coordinates of each pixel.")

        img_to_lab_to_5d = np.zeros((img_to_lab.shape[0], img_to_lab.shape[1], 5))
        for y, x in itertools.product(range(img_to_lab.shape[0]), range(img_to_lab.shape[1])):
                img_to_lab_to_5d[y, x] = np.append(img_to_lab[y, x], [y, x])

        processed_img = img_to_lab_to_5d
        processed_img_reshaped = processed_img.reshape((processed_img.shape[0] * processed_img.shape[1], processed_img.shape[2])).T

    print(f"Shape of the processed image is {processed_img_reshaped.shape}")
    print("\n")
    # run the meanshift algorithm
    start_time = time.time()
    if params.opt:
        print(f"Optimization is {params.opt}")
        labels, peaks = meanshift_opt(processed_img_reshaped, r=params.r, c=params.c)
    else:
        print(f"Optimization is {params.opt}")
        labels, peaks = meanshift(processed_img_reshaped, r=params.r)
    duration = time.time() - start_time
    print(f'The algorithm took {duration:.2f} s')

    segmented_img = np.zeros(processed_img_reshaped.T.shape)

    for i, _ in enumerate(processed_img_reshaped.T):
        segmented_img[i] = peaks[labels.astype(int)[i]]

    segmented_to_rgb = lab2rgb(segmented_img.reshape(processed_img.shape)[:, :, :3])

    num_peaks = peaks.shape[0]

    return img_to_rgb, segmented_to_rgb, processed_img_reshaped, labels, peaks, num_peaks


def plot_images(img_rgb, segmented_rgb, num_peaks, params):
    """
    Function that plots segmented and original images

    :param img_rgb: original image in the rgb color space
    :param segmented_rgb: segmented image in the rgb color space
    :param num_peaks: number of unique peaks that the algorithm found
    :param params: input parameters to the algorithm
    :return: plot a pair of images with titles
    """

    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(segmented_rgb)
    plt.title(f"r={params.r}, c={params.c},\nfeat.type={params.feature_type}, #of peaks = {num_peaks}", fontsize=20)
    plt.show()


if __name__ == '__main__':
    # parse an input
    parser = get_parser()
    params, unknown = parser.parse_known_args()

    # run the segmentation function
    img_rgb, segmented_rgb, processed_img_reshaped, labels, peaks, num_peaks = imageSegmentation(params)
    
    if params.save:
        # save segmented image
        mpimg.imsave("segmented_image.jpg", segmented_rgb)

    # display the segmented and original images
    plot_images(img_rgb, segmented_rgb, num_peaks, params)

    # plot a cloud of points
    if params.cloud:
        print("Printing a 3D cloud of points...")
        plotclusters3D(np.einsum("ij -> ji", processed_img_reshaped), labels, peaks)

