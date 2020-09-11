"""Set14 for single image super-resolution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@incollection{Zeyde2012,
  doi = {10.1007/978-3-642-27413-8_47},
  url = {https://doi.org/10.1007/978-3-642-27413-8_47},
  year = {2012},
  publisher = {Springer Berlin Heidelberg},
  pages = {711--730},
  author = {Roman Zeyde and Michael Elad and Matan Protter},
  title = {On Single Image Scale-Up Using Sparse-Representations},
  booktitle = {Curves and Surfaces}
}
"""

_DESCRIPTION = """
A set of 14 images to evaluate single image super-resolution.
"""

DOWNLOAD_URL = \
    "https://github.com/HedgehogCode/tensorflow-datasets-bw/releases/download/0.0.1rc/Set14.zip"


class Set14(tfds.core.GeneratorBasedBuilder):
    """Set14 for single image super-resolution."""

    VERSION = tfds.core.Version('0.3.0')

    def __init__(self, data_dir=None, config=None, version=None,
                 resize_method: str = tf.image.ResizeMethod.BICUBIC,
                 antialias: bool = False,
                 scale: int = 4) -> None:
        super(Set14, self).__init__(
            data_dir=data_dir, config=config, version=version)

        self.resize_method = resize_method
        self.antialias = antialias
        self.scale = scale

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'lr': tfds.features.Image(),
                'hr': tfds.features.Image()
            }),
            homepage='https://doi.org/10.1007/978-3-642-27413-8_47',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Download and extract
        resource = tfds.download.Resource(
            url=DOWNLOAD_URL, extract_method=tfds.download.ExtractMethod.ZIP)
        extracted_path = dl_manager.download_and_extract(resource)
        data_path = os.path.join(extracted_path, 'Set14')
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"images_dir_path": data_path},
            ),
        ]

    def _generate_examples(self, images_dir_path):
        """Yields examples."""
        for image_file in tf.io.gfile.listdir(images_dir_path):
            if image_file.endswith('png'):
                image_id = image_file[:-4]
                image_path = os.path.join(images_dir_path, image_file)
                yield image_id, {
                    'hr': image_path,
                    'lr': image_path
                }

    def _as_dataset(self, split=tfds.Split.TRAIN, decoders=None,
                    read_config=None, shuffle_files=False):
        dataset = super(Set14, self)._as_dataset(split=split,
                                                 decoders=decoders,
                                                 read_config=read_config,
                                                 shuffle_files=shuffle_files)

        def downsample(x):
            hr_shape = tf.shape(x['hr'])
            lr_size = (hr_shape[0] // self.scale,
                       hr_shape[1] // self.scale)
            hr_size = (lr_size[0] * self.scale,
                       lr_size[1] * self.scale)

            # Crop the high resolution image
            hr = x['lr'][:hr_size[0], :hr_size[1], :]

            # Resize the low resoltion image
            lr = tf.image.resize(x['lr'], size=lr_size,
                                 method=self.resize_method,
                                 antialias=self.antialias)
            # Clip values and back to uint8 (not needed for nearest neighbor interpolation)
            if not self.resize_method == 'nearest':
                lr = tf.cast(tf.clip_by_value(lr, 0, 255), tf.uint8)

            return {'hr': hr, 'lr': lr}

        return dataset.map(downsample)
