"""hci_lf dataset."""

import os
import re
import configparser

import numpy as np
import imageio
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
Data of the 4D Light Field Benchmark.

The dataset consists of
* 4 Stratified Scenes: `tfds.load("hci_lf/stratified", split="test")`
* 4 Test Scenes: `tfds.load("hci_lf", split="test")`
* 4 Training Scenes: `tfds.load("hci_lf", split="train")`
* 12 Additional Scenes: `tfds.load("hci_lf", split="validation")`
"""

_CITATION = """
@inproceedings{honauer2016dataset,
  title={A dataset and evaluation methodology for depth estimation on 4D light fields},
  author={Honauer, Katrin and Johannsen, Ole and Kondermann, Daniel and Goldluecke, Bastian},
  booktitle={Asian Conference on Computer Vision},
  pages={19--34},
  year={2016},
  organization={Springer}
}
"""


class HciLfConfig(tfds.core.BuilderConfig):
    def __init__(self, stratified=False, **kwargs):
        super(HciLfConfig, self).__init__(version=tfds.core.Version("0.1.0"), **kwargs)
        self.stratified = stratified


class HciLf(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for hci_lf dataset."""

    BUILDER_CONFIGS = [
        HciLfConfig(
            name="simulated",
            description="All simulated light fields",
            stratified=False,
        ),
        HciLfConfig(
            name="stratified",
            description="Stratisfied light fields",
            stratified=True,
        ),
    ]

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir (usually `~/tensorflow_datasets/downloads/manual`) should contain the folders
    * 'hcilf/additional'
    * 'hcilf/stratified'
    * 'hcilf/test'
    * 'hcilf/training'

    Request the dataset from https://lightfield-analysis.uni-konstanz.de/ and extract it.
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "lf": tfds.features.Tensor(
                        shape=(9, 9, 512, 512, 3), dtype=tf.uint8
                    ),
                    "depth": tfds.features.Tensor(shape=(512, 512), dtype=tf.float32),
                    "disparity": tfds.features.Tensor(
                        shape=(512, 512), dtype=tf.float32
                    ),
                }
            ),
            homepage="https://lightfield-analysis.uni-konstanz.de/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        data_path = os.path.join(dl_manager.manual_dir, "hcilf")

        if self.builder_config.stratified:
            return {
                "test": self._generate_examples(os.path.join(data_path, "stratified"))
            }

        return {
            "train": self._generate_examples(os.path.join(data_path, "training")),
            "test": self._generate_examples(os.path.join(data_path, "test")),
            "validation": self._generate_examples(
                os.path.join(data_path, "additional")
            ),
        }

    def _generate_examples(self, path):
        """Yields examples."""

        scenes = [n for n in tf.io.gfile.listdir(path) if n != "license.txt"]
        for scene in scenes:
            p = os.path.join(path, scene)

            # Read the important parameters
            config = configparser.ConfigParser()
            config.read(os.path.join(p, "parameters.cfg"))
            num_cams_y = int(config["extrinsics"]["num_cams_y"])
            num_cams_x = int(config["extrinsics"]["num_cams_x"])
            # num_cams_y = 9
            # num_cams_x = 9

            # Read the light field views
            view_paths = sorted(tf.io.gfile.glob(os.path.join(p, "input_*.png")))
            views = np.array([imageio.imread(v) for v in view_paths])
            lf = np.reshape(
                views, (num_cams_y, num_cams_x, views.shape[1], views.shape[2], 3)
            )

            # Read the depth map
            depth_file = os.path.join(p, "gt_depth_lowres.pfm")
            if tf.io.gfile.exists(depth_file):
                depth_map = _read_pfm(depth_file)
            else:
                depth_map = np.zeros([512, 512], dtype=np.float32)

            # Read the disparity map
            disp_file = os.path.join(p, "gt_disp_lowres.pfm")
            if tf.io.gfile.exists(disp_file):
                disp_map = _read_pfm(disp_file)
            else:
                disp_map = np.zeros([512, 512], dtype=np.float32)

            yield scene, {
                "lf": lf,
                "depth": depth_map,
                "disparity": disp_map,
            }


# Slightly adapted from https://gist.github.com/aminzabardast/cdddae35c367c611b6fd5efd5d63a326
def _read_pfm(file):
    """Read a PFM file into a Numpy array. Note that it will have
    a shape of H x W, not W x H. Returns a tuple containing the
    loaded image and the scale factor from the file.
    """
    with tf.io.gfile.GFile(file, "rb") as f:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = f.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.search(r"(\d+)\s(\d+)", f.readline().decode("ascii"))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(f.readline().rstrip())
        if scale < 0:  # little-endian
            endian = "<"
            scale = -scale
        else:
            endian = ">"  # big-endian

        data = np.frombuffer(f.read(), endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data * abs(scale)
