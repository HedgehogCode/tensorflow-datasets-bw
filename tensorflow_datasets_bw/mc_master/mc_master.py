"""The McMaster dataset for image demosaicking."""
import os
import glob
import imageio

import tensorflow_datasets as tfds

_DESCRIPTION = """
The McMaster dataset for color demosaicking (CDM) and color image processing.
"""

_CITATION = """
@article{zhang2011color,
  title={Color demosaicking by local directional interpolation and nonlocal adaptive thresholding},
  author={Zhang, Lei and Wu, Xiaolin and Buades, Antoni and Li, Xin},
  journal={Journal of Electronic imaging},
  volume={20},
  number={2},
  pages={023016},
  year={2011},
  publisher={International Society for Optics and Photonics}
}
"""


class McMaster(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mc_master dataset."""

    VERSION = tfds.core.Version('0.0.1')
    RELEASE_NOTES = {
        '0.0.1': 'Alpha release.',
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir (usually `~/tensorflow_datasets/downloads/manual`) should contain the folder
    'McM'. Download and extract the dataset from https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm.
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(None, None, 3)),
            }),
            homepage='https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        data_path = os.path.join(dl_manager.manual_dir, 'McM')

        return {
            'test': self._generate_examples(data_path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        files = glob.glob(os.path.join(path, '*.tif'))
        keys = [os.path.basename(n)[:-4] for n in files]
        for key, file in zip(keys, files):
            yield key, {
                'image': imageio.imread(file)
            }
