"""Data sets of the Multi-Dimensional Signal Processing Research Group (MDSP) for Video
Super-Resolution."""

import numpy as np
import scipy.io
import tensorflow_datasets as tfds

_DESCRIPTION = """
The data sets have been gathered during the past several years in the Multi-Dimensional Signal
Processing Research Group (MDSP).
"""

_CITATION = """
@misc{mdsp_milanfar,
  title={Peyman Milanfar},
  url={http://www.soe.ucsc.edu/~milanfar/software/sr-datasets.html},
  journal={MDSP Super-Resolution And Demosaicing Datasets :: Peyman Milanfar}
}
"""

NAMES = {
    "face_adyoron_1": "Color Face 1",
    "face_adyoron_2": "Color Face 2",
    "Adyoron_small": "Surveillance (Small)",
    "Book_case1_small": "Bookcase 1 (Small)",
    "Book_case1": "Bookcase 1",
}
DOWNLOAD_PATHS = {
    k: f"https://users.soe.ucsc.edu/~milanfar/software/datasets/{k}.mat" for k in NAMES.keys()
}


class MdspColorSr(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mdsp_color_sr dataset."""

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
                'video': tfds.features.Video(shape=(None, None, None, 3)),
            }),
            homepage='https://users.soe.ucsc.edu/~milanfar/software/sr-datasets.html',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        paths = dl_manager.download(DOWNLOAD_PATHS)

        return {
            'test': self._generate_examples(paths),
        }

    def _generate_examples(self, paths):
        """Yields examples."""
        for key, path in paths.items():
            video = scipy.io.loadmat(path)
            video = video[key]
            video = np.transpose(video, axes=[3, 0, 1, 2])
            print(video.dtype)
            yield NAMES[key], {
                'video': video
            }
