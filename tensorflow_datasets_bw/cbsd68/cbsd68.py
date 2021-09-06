"""The Berkeley dataset for contour detection and image segmentation."""

import os

import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Color BSD68 dataset for image denoising benchmarks.
It is part of The Berkeley Segmentation Dataset and
Benchmark https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
"""

_CITATION = """
@inproceedings{martin2001database,
  title={A database of human segmented natural images and its application to evaluating
        segmentation algorithms and measuring ecological statistics},
  author={Martin, David and Fowlkes, Charless and Tal, Doron and Malik, Jitendra},
  booktitle={Proceedings Eighth IEEE International Conference on Computer Vision. ICCV 2001},
  volume={2},
  pages={416--423},
  year={2001},
  organization={IEEE}
}
"""

DOWNLOAD_PATH = "https://github.com/clausmichele/CBSD68-dataset/archive/51a07a95884ac7c8bdd5d1614f9da781adc3c4a0.zip"

HOMEPAGE = 'https://github.com/clausmichele/CBSD68-dataset'

# TODO add configs for noise levels


class Cbsd68(tfds.core.GeneratorBasedBuilder):
    """The Berkeley dataset for contour detection and image segmentation."""

    VERSION = tfds.core.Version('0.0.1')
    RELEASE_NOTES = {
        '0.0.1': 'Alpha release.',
    }

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image()
                # TODO add noisy
            }),
            homepage=HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Download and extract
        extracted_path = dl_manager.download_and_extract(DOWNLOAD_PATH)
        data_path = os.path.join(
            extracted_path, 'CBSD68-dataset-51a07a95884ac7c8bdd5d1614f9da781adc3c4a0', 'CBSD68')
        original_path = os.path.join(data_path, 'original_png')
        return {
            'test': self._generate_examples(original_path),
        }

    def _generate_examples(self, original_path):
        """Yields examples."""
        files = sorted(tf.io.gfile.listdir(original_path))
        image_files = filter(lambda f: f.endswith('.png'), files)

        for image_file in image_files:
            image_id = image_file[:-4]
            yield image_id, {
                'image': os.path.join(original_path, image_file)
            }
