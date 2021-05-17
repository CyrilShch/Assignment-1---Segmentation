# imports
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
import matplotlib.image as mpimg
import argparse
import warnings
warnings.simplefilter("ignore")

# SCRIPT USAGE EXAMPLE:
# python Code/filtering.py --im Data/img1.jpg

def get_parser():
    """
    Function that gets parsed input parameters

    :return: parsed parameters
    """
    # parse parameters
    parser = argparse.ArgumentParser(description='Filtering image')
    parser.add_argument("--im", type=str, default="", help="Path to the image")

    return parser


def filtering(im):
    """
    Functions that takes an image and applies gaussian and median filters. It saves obtained images and displays the results.
    :param im: input image
    :return: plot with the original and filtered images.
    """
    # read the input image
    img = plt.imread(params.im)

    # apply filtering
    gauss = gaussian_filter(img, sigma=1)
    median = median_filter(img, size=3)

    # save filtered images
    mpimg.imsave("gauss_img.jpg", gauss)
    mpimg.imsave("median_img.jpg", median)

    # display images
    images = [img, gauss, median]
    names = ['Original', 'Gaussian filtering (sigma=1)', 'Median filtering']

    # Show subplots | shape: (1,3)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,12))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        plt.imshow(images[i])
        #plt.colorbar()
        plt.axis('off')
        plt.title(f'{names[i]}', fontsize=20 )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # parse an input
    parser = get_parser()
    params, unknown = parser.parse_known_args()

    # run the filtering function
    filtering(params)
