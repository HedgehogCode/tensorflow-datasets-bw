"""waterloo_exploration dataset."""

import os
import glob

import imageio
import tensorflow_datasets as tfds

_DESCRIPTION = """
The Waterloo Exploration database contains 4,744 pristine natural imageas and 94,880 distorted
images and was created to evaluate image quality assessment models.

In the current state this TensorFlow Dataset does only yield the pristine natural images. All
images are used for the 'train' split.
"""

_CITATION = """
@article{ma2016waterloo,
  title={Waterloo exploration database: New challenges for image quality assessment models},
  author={Ma, Kede and Duanmu, Zhengfang and Wu, Qingbo and Wang, Zhou and Yong, Hongwei and Li, Hongliang and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={2},
  pages={1004--1016},
  year={2016},
  publisher={IEEE}
}
"""  # noqa: E501

# TODO Get rid of imageio dependency and improve performance


class WaterlooExploration(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for waterloo_exploration dataset."""

    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {
        "0.0.1": "Alpha release.",
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir (usually `~/tensorflow_datasets/downloads/manual`) should contain the folder
    'exploration_database_and_code' download and extract the dataset from
    https://ece.uwaterloo.ca/~k29ma/exploration/.
      """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(None, None, 3)),
                }
            ),
            homepage="https://ece.uwaterloo.ca/~k29ma/exploration/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        data_path = os.path.join(
            dl_manager.manual_dir, "exploration_database_and_code", "pristine_images"
        )

        return {
            "train": self._generate_examples(data_path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        files = glob.glob(os.path.join(path, "*.bmp"))
        keys = [os.path.basename(n)[:-4] for n in files]
        for key, f in zip(keys, files):
            image = imageio.imread(f)
            if image.shape[2] != 3:
                # Some images contain an alpha channel
                # Remove it
                image = image[:, :, :3]
            yield key, {"image": image}
