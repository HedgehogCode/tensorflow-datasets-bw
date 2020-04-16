"""Convolution kernels from Schelten et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from scipy import io

_CITATION = """
@inproceedings{schelten_interleaved_2015,
  address = {Waikoloa, HI, USA},
  title = {Interleaved {Regression} {Tree} {Field} {Cascades} for {Blind} {Image} {Deconvolution}},
  isbn = {978-1-4799-6683-7},
  url = {http://ieeexplore.ieee.org/document/7045926/},
  doi = {10.1109/WACV.2015.72},
  urldate = {2020-01-25},
  booktitle = {2015 {IEEE} {Winter} {Conference} on {Applications} of {Computer} {Vision}},
  publisher = {IEEE},
  author = {Schelten, Kevin and Nowozin, Sebastian and Jancsary, Jeremy and Rother, Carsten and Roth, Stefan},
  month = jan,
  year = {2015},
  pages = {494--501},
}
"""  # noqa: E501

_DESCRIPTION = """
Realistic blur kernels from the paper Interleaved Regression Tree Field
Cascades for Blind Image Deconvolution by Kevin Schelten et al.
"""

DOWNLOAD_PATH = "https://bitbucket.org/visinf/projects-interleaved-rtf/raw/" + \
    "ae1f8558af8bbe09a55bdbe7bd64ed20d2c9f3fc/kernels.mat"

# The maximum height and width of the kernels
MAX_HEIGHT = 191
MAX_WIDTH = 145

DMSP_KERNEL_IDX = [19, 29, 67, 68, 95]


class ScheltenKernelsConfig(tfds.core.BuilderConfig):

    def __init__(self, dmsp_subset=False, **kwargs):
        super(ScheltenKernelsConfig, self).__init__(version=tfds.core.Version('0.2.0'), **kwargs)
        self.dmsp_subset = dmsp_subset


class ScheltenKernels(tfds.core.GeneratorBasedBuilder):
    """Realistic blur kernels from Schelten et al."""

    BUILDER_CONFIGS = [
        ScheltenKernelsConfig(name='all',
                              description="Use all kernels.",
                              dmsp_subset=False),
        ScheltenKernelsConfig(name='dmsp',
                              description="Use only the kernels used in the DMSP paper.",
                              dmsp_subset=True)
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'kernel':
                tfds.features.Tensor(
                    shape=[MAX_HEIGHT, MAX_WIDTH], dtype=tf.float64),
                'size': tfds.features.Tensor(shape=[2], dtype=tf.int32)
            }),
            homepage='https://bitbucket.org/visinf/projects-interleaved-rtf',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Download and extract
        dl_path = dl_manager.download(DOWNLOAD_PATH)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"dl_path": dl_path},
            ),
        ]

    def _generate_examples(self, dl_path):
        """Yields examples."""
        kernels = io.loadmat(dl_path)['kernels'][0]

        if self.builder_config.dmsp_subset:
            kernels = kernels[DMSP_KERNEL_IDX]

        for kernel_id, kernel in enumerate(kernels):
            # Pad the kernel to the max height and width
            size = kernel.shape
            padding = ((0, MAX_HEIGHT - size[0]),
                       (0, MAX_WIDTH - size[1]))
            kernel_padded = np.pad(kernel, padding)
            yield kernel_id, {
                'kernel': kernel_padded,
                'size': size
            }
