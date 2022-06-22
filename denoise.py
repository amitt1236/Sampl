import cv2 as cv
import util
import warnings
import numpy as np

def fastnl(process, intensity=10, show=False):
    """
    Denoise image using fastNlMeansDenoising

    :param process: image to denoise
    :param intensity: denoising intensity, higher value == less noise and more data loss
    :param show: True == show a figure with mask and original image
    :return: denoised image
    @author:Amit
    """
    process = cv.cvtColor(process, cv.COLOR_BGR2GRAY)
    denoised = cv.fastNlMeansDenoising(process, None, intensity, 5, 21)

    if show:
        util.im_show(process, denoised, "fastNl means denoising")

    return denoised


def fastnlMulti(process, intensity=10, show=False):
    """
    Takes a list of three consecutive frames with short duration between frames and
    uses similarly between frames to denoise. frame to denoised should be the middle frame.

    :param process: List of consecutive frames (needs to be short duration between frames)
    :param intensity: denoising intensity, higher value == less noise and more data loss
    :param show: True == show a figure with mask and original image, shows the middle frame
    :return: The middle frame in the list denoised
    @author:Amit
    """
    process = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in process]
    denoised = cv.fastNlMeansDenoisingMulti(process, 1, 1, None, intensity, 7, 35)

    if show:
        util.im_show(process[1], denoised, "fastNl means denoising multi")

    return denoised


def edgePreservingFilter(process,smooth=15, threshold=0.1, show=False):
    """
    Denoise image whole preserving edges between areas in the image

    :param process: image denoise
    :param smooth: denoising intensity, higher value == less noise and more data loss. range=(0-200)
    :param threshold: threshold to distinguished between areas. lower == preserve more edges data.'\ range=(0-1)
    :param show: True == show a figure with mask and original image, shows the middle frame
    :return: The middle frame in the list denoised
    @author:Amit
    """
    process = cv.cvtColor(process, cv.COLOR_BGR2GRAY)
    denoised = cv.edgePreservingFilter(process, None, 1, smooth, threshold)

    if show:
        util.im_show(process, denoised, "edge Preserving denoising")

    return denoised

def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),option=1,show=False):
    """
    Anisotropic diffusion.
    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)
    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration
    Returns:
            imgout   - diffused image.
    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.
    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)
    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes
    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.
    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.
    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>
    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>
    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    April 2019 - Corrected for Python 3.7 - AvW 
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for _ in range(niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaS/kappa)**2.)/step[0]
            gE = np.exp(-(deltaE/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaS/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

    if show:
        util.im_show(img, imgout, "anisotropicDiffusion denoising")

    return imgout


if __name__ == "__main__":
    img = cv.imread('examples/228.png')
    img1 = cv.imread('examples/336.png')
    img2 = cv.imread('examples/346.png')

    fastnl(img, show=True)
    fastnlMulti([img1, img2, img1], show=True)
    edgePreservingFilter(img, 15, 0.1, show=True)
    anisodiff(img, show=True)