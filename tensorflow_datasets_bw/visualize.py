import math
import matplotlib.pyplot as plt


def draw_dataset(dataset, num_images, num_col, figsize=(15, 40)):
    """Draw some images from a TensorFlow dataset.

    Args:
        dataset: The TensorFlow dataset.
        num_images: The number of images to draw.
        num_cols: The number of columns of the resulting figure.
        figsize: The size of the resulting figure (default: (15, 40)).
    """
    plt.figure(figsize=figsize)
    for idx, image in enumerate(dataset):
        plt.subplot(math.ceil(num_images / num_col), num_col, idx + 1)
        plt.imshow(image)
        plt.title(idx)
        if idx + 1 == num_images:
            break
    plt.show()


def draw_images(images, ncol=None, figsize=(15, 8)):
    """Draw the given images in one figure in one row.

    Args:
        images: A list/tensor of images or a list of tuples with names and images.
        ncol: The number of columns. None for as many columns as images. (default: None).
        figsize: The size of the resulting figure (default: (15, 8)).
    """
    if ncol is None:
        ncol = len(images)
        nrow = 1
    else:
        nrow = math.ceil(len(images) / ncol)

    fig, ax = plt.subplots(nrow, ncol, squeeze=False, figsize=figsize)

    for idx, image in enumerate(images):
        ridx = int(idx / ncol)
        cidx = idx % ncol

        if isinstance(image, tuple):
            ax[ridx, cidx].set_title(image[0])
            image = image[1]

        ax[ridx, cidx].imshow(image)
    plt.show()
