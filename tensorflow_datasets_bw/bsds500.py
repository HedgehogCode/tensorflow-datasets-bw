"""The Berkeley dataset for contour detection and image segmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@Article{amfm_pami2011,
 author = {Arbelaez, Pablo and Maire, Michael and Fowlkes, Charless and Malik, Jitendra},
 title = {Contour Detection and Hierarchical Image Segmentation},
 journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
 issue_date = {May 2011},
 volume = {33},
 number = {5},
 month = may,
 year = {2011},
 issn = {0162-8828},
 pages = {898--916},
 numpages = {19},
 url = {http://dx.doi.org/10.1109/TPAMI.2010.161},
 doi = {10.1109/TPAMI.2010.161},
 acmid = {1963088},
 publisher = {IEEE Computer Society},
 address = {Washington, DC, USA},
 keywords = {Contour detection, image segmentation, computer vision.},
}
"""

_DESCRIPTION = """
The goal of this work is to provide an empirical basis for research on image
segmentation and boundary detection. In order to promote scientific progress in
the study of visual grouping, we provide the following resources:

* A large dataset of natural images that have been manually segmented. The
  human annotations serve as ground truth for learning grouping cues as well as
  a benchmark for comparing different segmentation and boundary detection
  algorithms.
* The most recent algorithms our group has developed for contour detection and
  image segmentation.
* Performance evaluation of the leading computational approaches to grouping.
"""

DOWNLOAD_PATH = \
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz"

HOMEPAGE = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html'

DMSP_IMAGES_IDS = ['101085', '108070', '130026', '148089', '167083',
                   '196073', '220075', '241048', '295087', '304074']


class Bsds500Config(tfds.core.BuilderConfig):

    def __init__(self, dmsp_subset=False, **kwargs):
        super(Bsds500Config, self).__init__(version=tfds.core.Version('0.2.0'), **kwargs)
        self.dmsp_subset = dmsp_subset


class Bsds500(tfds.core.GeneratorBasedBuilder):
    """The Berkeley dataset for contour detection and image segmentation."""

    BUILDER_CONFIGS = [
        Bsds500Config(name='all',
                      description="Use all kernels.",
                      dmsp_subset=False),
        Bsds500Config(name='dmsp',
                      description="Use only the images used in the DMSP paper.",
                      dmsp_subset=True)
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image()
                # TODO add segmentation and contour ground truth
            }),
            homepage=HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Download and extract
        extracted_path = dl_manager.download_and_extract(DOWNLOAD_PATH)
        data_path = os.path.join(extracted_path, 'BSR', 'BSDS500', 'data')
        images_path = os.path.join(data_path, 'images')
        if self.builder_config.dmsp_subset:
            return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.VALIDATION,
                    gen_kwargs={
                        "images_dir_path": os.path.join(images_path, 'val')
                    },
                ),
            ]
        else:
            return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TRAIN,
                    gen_kwargs={
                        "images_dir_path": os.path.join(images_path, 'train')
                    },
                ),
                tfds.core.SplitGenerator(
                    name=tfds.Split.VALIDATION,
                    gen_kwargs={
                        "images_dir_path": os.path.join(images_path, 'val')
                    },
                ),
                tfds.core.SplitGenerator(
                    name=tfds.Split.TEST,
                    gen_kwargs={
                        "images_dir_path": os.path.join(images_path, 'test')
                    },
                ),
            ]

    def _generate_examples(self, images_dir_path):
        """Yields examples."""
        files = sorted(tf.io.gfile.listdir(images_dir_path))
        image_files = filter(lambda f: f.endswith('.jpg'), files)

        for image_file in image_files:
            image_id = image_file[:-4]
            if not self.builder_config.dmsp_subset or image_id in DMSP_IMAGES_IDS:
                yield image_id, {
                    'image': os.path.join(images_dir_path, image_file)
                }
