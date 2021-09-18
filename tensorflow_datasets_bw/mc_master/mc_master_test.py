"""The McMaster dataset for image demosaicking."""

import tensorflow_datasets as tfds
from . import mc_master


class McMasterTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for mc_master dataset."""

    DATASET_CLASS = mc_master.McMaster
    SPLITS = {
        "test": 2,  # Number of fake test example
    }
    DL_EXTRACT_RESULT = "McM"


if __name__ == "__main__":
    tfds.testing.test_main()
