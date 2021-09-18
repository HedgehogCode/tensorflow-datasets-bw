"""vid4 dataset."""

import os

import imageio
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Classical dataset for testing video super-resolution consisting of 4 image sequences.

The sequences are
* walk (740x480, 47 frames)
* foliage (740x480, 49 frames)
* city (704x576, 34 frames)
* calendar (720x576, 41 frames)
"""

_CITATION = """
@inproceedings{liu2011bayesian,
  title={A bayesian approach to adaptive video super resolution},
  author={Liu, Ce and Sun, Deqing},
  booktitle={CVPR 2011},
  pages={209--216},
  year={2011},
  organization={IEEE}
}
"""

DOWNLOAD_URL = "https://github.com/HedgehogCode/tensorflow-datasets-bw/releases/download/0.10.0/Vid4.zip"


class Vid4(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for vid4 dataset."""

    VERSION = tfds.core.Version("0.1.0")
    RELEASE_NOTES = {
        "0.1.0": "Initial release.",
    }

    def __init__(
        self,
        data_dir=None,
        config=None,
        version=None,
        resize_method: str = tf.image.ResizeMethod.BICUBIC,
        antialias: bool = False,
        scale: int = 4,
    ) -> None:
        super(Vid4, self).__init__(data_dir=data_dir, config=config, version=version)
        self.resize_method = resize_method
        self.antialias = antialias
        self.scale = scale

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "hr": tfds.features.Video(shape=(None, None, None, 3)),
                    "lr": tfds.features.Video(shape=(None, None, None, 3)),
                }
            ),
            homepage="https://github.com/YounggjuuChoi/Deep-Video-Super-Resolution/blob/master/Doc/Dataset.md#1-vid4",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(DOWNLOAD_URL)

        return {
            "test": self._generate_examples(os.path.join(path, "Vid4")),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        for sequence in ["walk", "foliage", "city", "calendar"]:
            with tf.io.gfile.GFile(os.path.join(path, f"{sequence}.txt"), "r") as f:
                image_files = f.read().splitlines()
            video = np.array(
                [imageio.imread(os.path.join(path, f)) for f in image_files]
            )
            yield sequence, {"hr": video, "lr": np.zeros((1, 1, 1, 3), dtype=np.uint8)}

    def _as_dataset(
        self, split="train", decoders=None, read_config=None, shuffle_files=False
    ):
        dataset = super(Vid4, self)._as_dataset(
            split=split,
            decoders=decoders,
            read_config=read_config,
            shuffle_files=shuffle_files,
        )

        def downsample(x):
            video = x["hr"]  # x['hr'] and x['lr'] are equal
            hr_shape = tf.shape(video)
            lr_size = (hr_shape[1] // self.scale, hr_shape[2] // self.scale)
            hr_size = (lr_size[0] * self.scale, lr_size[1] * self.scale)

            # Crop the high resolution video
            hr = video[:, : hr_size[0], : hr_size[1], :]

            # Resize the low resoltion image
            lr = tf.image.resize(
                hr, size=lr_size, method=self.resize_method, antialias=self.antialias
            )
            # Clip values and back to uint8 (not needed for nearest neighbor interpolation)
            if not self.resize_method == "nearest":
                lr = tf.round(lr)
                lr = tf.clip_by_value(lr, 0, 255)
                lr = tf.cast(lr, tf.uint8)

            return {"hr": hr, "lr": lr}

        return dataset.map(downsample)
