"""A wrapper for the script tensorflow_datasets.scripts.download_and_prepare which
imports tensorflow_datasets_bw."""
from absl import app
import tensorflow_datasets.scripts.download_and_prepare as script
import tensorflow_datasets_bw  # noqa: F401

if __name__ == '__main__':
    app.run(script.main)
