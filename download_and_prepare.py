"""A wrapper for the script tensorflow_datasets.scripts.download_and_prepare which
imports tensorflow_datasets_bw."""
import subprocess

from absl import app
import tensorflow_datasets.scripts.download_and_prepare as script
import tensorflow_datasets_bw

if __name__ == '__main__':
    app.run(script.main)