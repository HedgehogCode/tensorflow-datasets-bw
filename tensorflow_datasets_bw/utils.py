"""Utilities to handle datasets like common map functions.
"""
from typing import Any, List, Tuple, TypeVar, Dict, Callable, Union

import tensorflow as tf

T = TypeVar('T')
K = TypeVar('K')


def get_one_example(dataset: tf.data.Dataset, index: int = 0,
                    random: bool = False):
    """Get one example of a TensorFlow dataset for testing/visualization.

    Args:
        dataset: The TensorFlow dataset.
        index: The index of the example (default: 0).
        random: If the dataset should be shuffled before the example is drawn
            (default: False).

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


# General helpers for mapping functions

def compose(*functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose the given functions which all take one argument.

    Args:
        functions: The functions to compose

    Returns:
        A function which applies the given functions in the order they appear in the argument
    """
    def apply(x):
        for f in functions:
            x = f(x)
        return x
    return apply


def map_on_dict(map: Callable[[tf.Tensor], tf.Tensor]
                ) -> Callable[[Dict[K, tf.Tensor]], Dict[K, tf.Tensor]]:
    """Apply the mapping function on each element of a dict.

    Args:
        map: The mapping function.

    Retuns:
        A function which applies the mapping function on each element of a
        dict.
    """
    def apply(x: Dict[K, tf.Tensor]) -> Dict[K, tf.Tensor]:
        return {k: map(v) for k, v in x.items()}
    return apply


def map_on_dict_key(key: K, mapping: Callable[[T], T]
                    ) -> Callable[[Dict[K, T]], Dict[K, T]]:
    """Apply a mapping function on one key of the dict and leave the others unchanged.

    Args:
        key: The key in the dictionary
        mapping: A function mapping the tensor with this key to a new value

    Returns:
        A function which takes a dictionary and applies the given function on the one
        element.
    """
    def apply(d):
        d[key] = mapping(d[key])
        return d
    return apply


def get_value(key: K) -> Callable[[Dict[K, T]], T]:
    """Get the object with the given key from the dict.

    Args:
        key: The key.

    Returns:
        Returns a function which gets the value with the given key from a
        given dict.
    """
    def get(x: Dict[K, T]) -> T:
        return x[key]
    return get


# Helpers for mapping images

def get_image(x: Dict[str, T]) -> T:
    """Get the object with the key 'image' from the dict.

    Args:
        x: The dict.

    Returns:
        The value for the key 'image'.
    """
    return x['image']


def to_float32(x: tf.Tensor) -> tf.Tensor:
    """Cast the given tensor to float32.

    Args:
        x: The tensor of any type.

    Returns:
        The tensor casts to float32.
    """
    return tf.cast(x, tf.float32)


def from_255_to_1_range(x: tf.Tensor) -> tf.Tensor:
    """Change the range of the tensor from 0-255 to 0-1.

    Args:
        x: The tensor of range 0-255.

    Returns:
        The tensor in range 0-1 (by dividing by 255).
    """
    return x / 255


def resize(size: Union[List[int], Tuple[int], tf.TensorShape]
           ) -> Callable[[tf.Tensor], tf.Tensor]:
    """Create a function which resizes an image tensor to the given size.

    Args:
        size: The target size of the image tensor.

    Returns:
        A function which takes an image tensor as argument and returns a
        resized image tensor.
    """
    def apply(x):
        return tf.image.resize(x, size)

    return apply


def crop_kernel_to_size(x: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Crops the tensor with the name 'kernel' to the original size.


    Args:
        x: A dictionary which should contain a tensor for the key 'kernel which will be cropped and
            a tensor for the key 'size' which defines the size of the cropped tensor.

    Returns:
        A dictionary with the cropped tensor for the element with the key 'kernel'.
    """
    kernel = x['kernel']
    size = x['size']
    return {'kernel': kernel[:size[0], :size[1]]}


# Helpers for light fields

def lf_to_batch(lf: tf.Tensor) -> tf.Tensor:
    """Flatten the first two dimensions of a light field to get a batch of images.

    Args:
        lf (TensorLike): The light field with shape [GH, GW, H, W, C]

    Returns:
        tf.Tensor: The flattened batch of light field images with shape [GH x GW, H, W, C]
    """
    size = tf.shape(lf)[2:]
    return tf.reshape(lf, (-1, *size))


def lf_batch_idx(grid, i: int, j: int):
    """Compute the index of a scpecific frame in the flattened light field.

    Args:
        grid (Tuple): The shape of the light field grid
        i: The index in the first grid dimension
        j: The index in the second grid dimension

    Returns:
        The index in a flattened light field.
    """
    return i * grid[1] + j
