"""Set5 for single image super-resolution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@inproceedings{Bevilacqua2012,
  doi = {10.5244/c.26.135},
  url = {https://doi.org/10.5244/c.26.135},
  year = {2012},
  publisher = {British Machine Vision Association},
  author = {Marco Bevilacqua and Aline Roumy and Christine Guillemot and Marie-line Alberi Morel},
  title = {Low-Complexity Single-Image Super-Resolution based on Nonnegative Neighbor Embedding},
  booktitle = {Procedings of the British Machine Vision Conference 2012}
}
"""

_DESCRIPTION = """
A set of 5 images to evaluate single image super-resolution.
"""

DOWNLOAD_PATH = "https://github.com/HedgehogCode/tensorflow-datasets-bw/releases/download/0.0.1rc/Set5.zip"

_DATA_OPTIONS = ['bicubic_x4']


class Set5Config(tfds.core.BuilderConfig):
    """BuilderConfig for Set5"""
    def __init__(self, name, **kwargs):
        if name not in _DATA_OPTIONS:
            raise ValueError("data must be one of %s" % _DATA_OPTIONS)

        description = kwargs.get("description", "Uses %s data." % name)
        kwargs["description"] = description

        super(Set5Config, self).__init__(name=name, **kwargs)
        self.data = name


def _make_builder_configs():
    def config_for(n):
        return Set5Config(version=tfds.core.Version('0.1.0'), name=n)

    return [config_for(n) for n in _DATA_OPTIONS]


class Set5(tfds.core.GeneratorBasedBuilder):
    """Set5 for single image super-resolution."""

    BUILDER_CONFIGS = _make_builder_configs()

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # 'lr': tfds.features.Image(),
                'hr': tfds.features.Image()
            }),
            homepage=
            'http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Download and extract
        extracted_path = dl_manager.extract(dl_manager.download(DOWNLOAD_PATH))
        data_path = os.path.join(extracted_path, 'Set5')
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
                yield image_id, {
                    'hr': os.path.join(images_dir_path, image_file)
                }
