import os
import tensorflow_datasets as tfds
from .bsds500 import Bsds500
from .schelten_kernels import ScheltenKernels
from .set5 import Set5
from .set14 import Set14
from .mdsp_color_sr import MDSPColorSR
from .utils import *  # noqa: F403

tfds.download.add_checksums_dir(
    os.path.join(os.path.abspath(__file__), '..', 'url_checksums'))
