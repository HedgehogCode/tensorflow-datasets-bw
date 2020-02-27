"""Utilities to handle datasets like common map functions.
"""
import tensorflow as tf


def get_image(x):
    """Get the object with the key 'image' from the dict.

    Args:
        x: The dict.
    
    Returns:
        The value for the key 'image'.
    """
    return x['image']


def to_float32(x):
    """Cast the given tensor to float32.

    Args:
        x: The tensor of any type.
    
    Returns:
        The tensor casts to float32.
    """
    return tf.cast(x, tf.float32)


def from_255_to_1_range(x):
    """Change the range of the tensor from 0-255 to 0-1.

    Args:
        x: The tensor of range 0-255.

    Returns:
        The tensor in range 0-1 (by dividing by 255).
    """
    return x / 255


def resize(size):
    """Create a function which resizes an image tensor to the given size.

    Args:
        size: The target size of the image tensor.

    Returns:
        A function which takes an image tensor as argument and returns a resized image tensor.
    """
    def apply(x):
        return tf.image.resize(x, size)

    return apply


def get_one_example(dataset, index=0, random=False):
    """Get one example of a TensorFlow dataset for testing/visualization.

    Args:
        dataset: The TensorFlow dataset.
        index: The index of the example (default: 0).
        random: If the dataset should be shuffled before the example is drawn (default: False).
    
    Returns:
        One example tensor of the given dataset.
    """
    if random:
        d = dataset.shuffle(100)
    else:
        d = dataset

    i = 0
    for x in d:
        if i == index:
            return x
        i += 1
