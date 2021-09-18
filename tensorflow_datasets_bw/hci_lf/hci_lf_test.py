"""hci_lf dataset."""

import tensorflow_datasets as tfds
from . import hci_lf


class HciLfTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for hci_lf dataset."""

    DATASET_CLASS = hci_lf.HciLf
    BUILDER_CONFIG_NAMES_TO_TEST = ["simulated"]
    SPLITS = {
        "train": 1,
        "test": 1,
        "validation": 1,
    }
    DL_EXTRACT_RESULT = "hcilf"


if __name__ == "__main__":
    tfds.testing.test_main()
