"""The Kodak Lossless True Color Image Suite."""

import tensorflow_datasets as tfds
from . import kodak24


class Kodak24Test(tfds.testing.DatasetBuilderTestCase):
    """Tests for kodak24 dataset."""
    DATASET_CLASS = kodak24.Kodak24
    SPLITS = {
        'test': 1,  # Number of fake test example
    }
    DL_EXTRACT_RESULT = {'01': 'kodim01.png'}


if __name__ == '__main__':
    tfds.testing.test_main()
