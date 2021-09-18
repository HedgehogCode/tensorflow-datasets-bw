"""set14 dataset."""

import tensorflow_datasets as tfds
from . import set14


class Set14Test(tfds.testing.DatasetBuilderTestCase):
    """Tests for set14 dataset."""

    DATASET_CLASS = set14.Set14
    SPLITS = {
        "test": 2,  # Number of fake test example
    }
    DL_EXTRACT_RESULT = "."


if __name__ == "__main__":
    tfds.testing.test_main()
