import os
import tensorflow_datasets as tfds
from .bsds500 import Bsds500
from .schelten_kernels import ScheltenKernels
from .utils import *

tfds.download.add_checksums_dir(
    os.path.join(os.path.abspath(__file__), '..', 'url_checksums'))
