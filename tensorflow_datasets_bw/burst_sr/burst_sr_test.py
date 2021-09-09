"""The BurstSR dataset for multi-frame super-resolution."""

import tensorflow_datasets as tfds
from . import burst_sr


class BurstSrTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for burst_sr dataset."""
    DATASET_CLASS = burst_sr.BurstSr
    SPLITS = {
        'validation': 2,  # Number of fake test example
    }
    DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
    tfds.testing.test_main()
