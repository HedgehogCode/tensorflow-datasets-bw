import math
import tensorflow as tf
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


def draw_images(images, figsize=(15, 8)):
    """Draw the given images in one figure in one row.

    Args:
        images: A list/tensor of images.
        figsize: The size of the resulting figure (default: (15, 8)).
    """
    plt.figure(figsize=figsize)
    for idx, i in enumerate(images):
        plt.subplot(1, len(images), idx + 1)
        plt.imshow(i)
    plt.show()