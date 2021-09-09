"""mdsp_color_sr dataset."""

import tensorflow_datasets as tfds
from . import mdsp_color_sr


class MdspColorSrTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for mdsp_color_sr dataset."""
    DATASET_CLASS = mdsp_color_sr.MdspColorSr
    SPLITS = {
        'test': 1,
    }
    DL_EXTRACT_RESULT = {'face_adyoron_1': 'face_adyoron_1.mat'}


if __name__ == '__main__':
    tfds.testing.test_main()
