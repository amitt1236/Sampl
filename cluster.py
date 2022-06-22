import cv2 as cv
import numpy as np
import util


def k_means_cluster(process, k, c_iter, show=False, chose=-1):
    """
    k means clustering algorithm

    :param process:  image to cluster
    :param k: number of clusters
    :param c_iter: number of iteration for the clustering algorithm
    :param show: True == show a figure with clustered image and original image
    :param chose: cluster to disable (will show as black)
    :return: clustered image
    @author:Amit
    """
    Z = process.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, c_iter, 1.0)  # `( type, max_iter, epsilon )`
    ret, label, center = cv.kmeans(Z, k, None, criteria, c_iter, cv.KMEANS_RANDOM_CENTERS)
    labels = label.flatten()
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((process.shape))

    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(res2)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    if chose != -1:
        masked_image[labels == chose] = [0, 0, 0]
    # convert back to original shape
    masked_image = masked_image.reshape(res2.shape)
    if show:
        util.im_show(process, masked_image, "k means cluster")
    return masked_image


if __name__ == "__main__":
    img = cv.imread('examples/228.png')
    k_means_cluster(img, 3, 20, show=True)
