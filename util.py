import matplotlib.pyplot as plt


def im_show(im1, im2, title):
    """
    Display two image in a figure

    :param im1: first image
    :param im2: second image
    :param title: Window name
    @author:Amit
    """
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(im1, cmap='gray')
    axes[1].imshow(im2, cmap='gray')
    fig.canvas.manager.set_window_title(title)
    fig.suptitle(title)
    plt.show()
