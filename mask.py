import cv2 as cv
import numpy as np
import util


def biggest_contour(process, approx=0.005, show=False):
    """
    Calculates the largest contour in input image and creates a binary
    mask for that contour. Tested with US sonogram

    :param process: image
    :param approx: mask edges approximation (lower == tighter approximation)
    :param show: True == show a figure with mask and original image
    :return: Binary mask
    @author:Amit
    """
    # Image process
    process = cv.cvtColor(process, cv.COLOR_BGR2GRAY)
    process = cv.GaussianBlur(process, (5, 5), 0)
    # Finding contours
    contours, hierarchy = cv.findContours(process, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    perimeter = max(contours, key=cv.contourArea)
    peri = cv.arcLength(perimeter, True)
    perimeter = cv.approxPolyDP(perimeter, approx * peri, True)
    # Create mask
    mask = np.zeros(process.shape)
    cv.fillPoly(mask, [perimeter], 1)
    mask = mask.astype(bool)

    if show:
        util.im_show(process, mask, "mask")

    return mask


if __name__ == "__main__":
    img = cv.imread('examples/228.png')
    biggest_contour(img, show=True)