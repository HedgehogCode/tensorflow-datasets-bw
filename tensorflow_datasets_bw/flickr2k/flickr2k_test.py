"""flickr2k dataset."""

import tensorflow_datasets as tfds
from . import flickr2k


class Flickr2kTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for flickr2k dataset."""
    DATASET_CLASS = flickr2k.Flickr2k
    SPLITS = {
        'train': 3,  # Number of fake train example
    }
    DL_EXTRACT_RESULT = 'Flickr2k'


if __name__ == '__main__':
    tfds.testing.test_main()
