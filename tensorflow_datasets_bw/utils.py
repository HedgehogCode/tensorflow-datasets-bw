"""Utilities to handle datasets like common map functions.
"""
from typing import List, Tuple, TypeVar, Dict, Callable, Union

import tensorflow as tf

T = TypeVar('T')
K = TypeVar('K')


def get_image(x: Dict[str, T]) -> T:
    """Get the object with the key 'image' from the dict.

    Args:
        x: The dict.

    Returns:
        The value for the key 'image'.
    """
    return x['image']


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


def to_float32(x: tf.Tensor) -> tf.Tensor:
    """Cast the given tensor to float32.

    Args:
        x: The tensor of any type.

    Returns:
        The tensor casts to float32.
    """
    return tf.cast(x, tf.float32)


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
