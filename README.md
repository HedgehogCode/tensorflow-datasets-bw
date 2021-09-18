# TensorFlow Datasets BW

[![Build Status](https://travis-ci.org/HedgehogCode/tensorflow-datasets-bw.svg?branch=master)](https://travis-ci.org/HedgehogCode/tensorflow-datasets-bw)

A collection of datasets extending tensorflow-datasets.

## Running tests

```
$ pytest
```

## Adding a checksum

```
export PYTHONPATH=$(pwd)
tfds build --imports tensorflow_datasets_bw --register_checksums cbsd68
```

## TODOs

* Add a License
* Improve the README.md
* Add segmentation to bsds500 dataset
* Add datasets