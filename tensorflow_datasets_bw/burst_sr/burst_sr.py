"""The BurstSR dataset for multi-frame super-resolution."""

import os
import glob
import imageio
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
The BurstSR dataset contains RAW bursts captured from a Samsung Galaxy S8 and corresponding HR
ground truths captured using a DSLR camera.
"""

_CITATION = """
@inproceedings{bhat2021deep,
  title={Deep burst super-resolution},
  author={Bhat, Goutam and Danelljan, Martin and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9209--9218},
  year={2021}
}
"""


# TODO add the training split
class BurstSr(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for burst_sr dataset."""

    VERSION = tfds.core.Version('0.0.1')
    RELEASE_NOTES = {
        '0.0.1': 'Alpha release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'hr': tfds.features.Image(shape=(None, None, 3), dtype=tf.uint16),
                'lr': tfds.features.Video(shape=(None, None, None, 3), dtype=tf.uint16),
            }),
            homepage='https://github.com/goutamgmb/deep-burst-sr',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        val_download_path = dl_manager.download_and_extract(
            'https://data.vision.ee.ethz.ch/bhatg/BurstSRChallenge/val.zip')

        val_path = os.path.join(val_download_path, "val")

        return {
            'validation': self._generate_examples([val_path]),
        }

    def _generate_examples(self, paths):
        """Yields examples."""
        imageio.plugins.freeimage.download()
        im_name = "im_raw.png"
        for path in paths:
            for key in os.listdir(path):
                this_path = os.path.join(path, key)
                hr = imageio.imread(os.path.join(this_path, "canon", im_name), format="PNG-FI")
                lr_paths = [
                    os.path.join(x, im_name) for x in glob.glob(os.path.join(this_path, "samsung_*"))
                ]
                lr = [imageio.imread(p, format="PNG-FI") for p in lr_paths]
                yield key, {
                    'hr': hr,
                    'lr': lr,
                }
