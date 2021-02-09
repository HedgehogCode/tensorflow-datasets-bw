"""flickr2k dataset."""

import os
import glob

import tensorflow_datasets as tfds

_DESCRIPTION = """
The Flickr2k dataset was collected using the Flickr API. It contains 2650 images. Each image is
available in high resolution, low resolution with bicubic downsampling and low resolution with
unknown downsampling. Downsamling factors are 2x, 3x and 4x.

In the current release the dataset only provides the high resolution images.

The dataset only consists of a 'train' split.
"""

_CITATION = """
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
"""

# TODO add LR features


class Flickr2k(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for flickr2k dataset."""

    VERSION = tfds.core.Version('0.0.1')
    RELEASE_NOTES = {
        '0.0.1': 'Alpha release.',
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir (usually `~/tensorflow_datasets/downloads/manual`) should contain the folder
    'Flickr2k' download and extract the dataset from
    https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar.
      """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'hr': tfds.features.Image(shape=(None, None, 3)),
            }),
            homepage='https://github.com/limbee/NTIRE2017',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        data_path = os.path.join(dl_manager.manual_dir,
                                 'Flickr2k', 'Flickr2k_HR')

        return {
            'train': self._generate_examples(data_path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        files = glob.glob(os.path.join(path, '*.png'))
        keys = [os.path.basename(n)[:-4] for n in files]
        for key, file in zip(keys, files):
            yield key, {
                'hr': file
            }
