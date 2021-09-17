"""vid4 dataset."""

import tensorflow_datasets as tfds
from . import vid4


class Vid4Test(tfds.testing.DatasetBuilderTestCase):
    """Tests for vid4 dataset."""
    DATASET_CLASS = vid4.Vid4
    SPLITS = {
        'test': 4,  # Number of fake test example
    }

    DL_EXTRACT_RESULT = "."


if __name__ == '__main__':
    tfds.testing.test_main()
