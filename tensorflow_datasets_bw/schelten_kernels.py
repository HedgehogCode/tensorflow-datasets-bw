"""Convolution kernels from Schelten et al."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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
"""

_DESCRIPTION = """
Realistic blur kernels from the paper Interleaved Regression Tree Field
Cascades for Blind Image Deconvolution by Kevin Schelten et al.
"""

DOWNLOAD_PATH = "https://bitbucket.org/visinf/projects-interleaved-rtf/raw/ae1f8558af8bbe09a55bdbe7bd64ed20d2c9f3fc/kernels.mat"


class ScheltenKernels(tfds.core.GeneratorBasedBuilder):
    """Realistic blur kernels from Schelten et al."""

    VERSION = tfds.core.Version('0.1.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'kernel':
                tfds.features.Tensor(shape=[1, None, None], dtype=tf.float64)
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
        for kernel_id, kernel in enumerate(kernels):
            print("Kernel shape", kernel.shape)
            yield kernel_id, {'kernel': kernel[None, ...]}


def get_kernels_dataset():
    """Load the schelten kernels. HACK: Until the datasets integration works"""
    def generator():
        for k in get_kernels_list():
            yield k

    return tf.data.Dataset.from_generator(generator, output_types=tf.float64)


def get_kernels_list():
    """Load the schelten kernels. HACK: Until the datasets integration works"""

    file_path = os.path.join(os.path.abspath(__file__), '..', 'kernels.mat')
    file_path = os.path.normpath(file_path)
    kernels = io.loadmat(file_path)['kernels'][0]
    return [k for k in kernels]