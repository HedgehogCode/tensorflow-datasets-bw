"""Data sets of the Multi-Dimensional Signal Processing Research Group (MDSP) for Video
Super-Resolution."""

import tensorflow_datasets as tfds
import scipy.io
import numpy as np

_DESCRIPTION = """
The data sets have been gathered during the past several years in the Multi-Dimensional Signal
Processing Research Group (MDSP).

The data point consist of a name and an RGB video.
"""

_CITATION = """
@misc{mdsp_milanfar,
  title={Peyman Milanfar},
  url={http://www.soe.ucsc.edu/~milanfar/software/sr-datasets.html},
  journal={MDSP Super-Resolution And Demosaicing Datasets :: Peyman Milanfar}
}
"""

DOWNLOAD_PATHS = [
    "https://users.soe.ucsc.edu/~milanfar/software/datasets/face_adyoron_1.mat",
    "https://users.soe.ucsc.edu/~milanfar/software/datasets/face_adyoron_2.mat",
    "https://users.soe.ucsc.edu/~milanfar/software/datasets/Adyoron_small.mat",
    "https://users.soe.ucsc.edu/~milanfar/software/datasets/Book_case1_small.mat",
    "https://users.soe.ucsc.edu/~milanfar/software/datasets/Book_case1.mat"
]
NAMES = [
    ("Color Face 1", "face_adyoron_1"),
    ("Color Face 2", "face_adyoron_2"),
    ("Surveillance (Small)", "Adyoron_small"),
    ("Bookcase 1 (Small)", "Book_case1_small"),
    ("Bookcase 1", "Book_case1")
]

HOMEPAGE = 'https://users.soe.ucsc.edu/~milanfar/software/sr-datasets.html'


class MdspColorSr(tfds.core.GeneratorBasedBuilder):
    """Data sets of the Multi-Dimensional Signal Processing Research Group (MDSP) for Video
    Super-Resolution."""

    VERSION = tfds.core.Version('0.1.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'name': tfds.features.Text(),
                'video': tfds.features.Video(shape=(None, None, None, 3))
            }),
            homepage=HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Download and extract
        video_paths = dl_manager.download(DOWNLOAD_PATHS)
        # for download_url in DOWNLOAD_PATHS:
        #     video_paths.append(dl_manager.download(download_url))
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"video_paths": video_paths}
            ),
        ]

    def _generate_examples(self, video_paths):
        """Yields examples."""
        for idx, ((name, key), video_path) in enumerate(zip(NAMES, video_paths)):
            video = scipy.io.loadmat(video_path)[key]
            video = np.transpose(video, axes=[3, 0, 1, 2])
            yield idx, {
                'name': name,
                'video': video
            }
