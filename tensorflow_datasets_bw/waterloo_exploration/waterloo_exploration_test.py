"""waterloo_exploration dataset."""

import tensorflow_datasets as tfds
from . import waterloo_exploration


class WaterlooExplorationTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for waterloo_exploration dataset."""

    DATASET_CLASS = waterloo_exploration.WaterlooExploration
    SPLITS = {
        "train": 3,  # Number of fake train example
    }
    DL_EXTRACT_RESULT = "exploration_database_and_code"


if __name__ == "__main__":
    tfds.testing.test_main()
