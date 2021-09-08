"""The Kodak Lossless True Color Image Suite."""

import tensorflow_datasets as tfds

_DESCRIPTION = """
Lossless, true color images. Released by the Eastman Kodak Company for unrestricted usage.
Commonly used for compression and denoising testing.
"""

_CITATION = """
@misc{franzen,
  title={ Kodak Lossless True Color Image Suite},
  url={http://r0k.us/graphics/kodak/},
  journal={True Color Kodak Images},
  author={Franzen, Richard}
}
"""

IMAGE_URLS = {
    f"{i:02d}": f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png" for i in range(1, 25)
}


class Kodak24(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for kodak24 dataset."""

    VERSION = tfds.core.Version('0.0.1')
    RELEASE_NOTES = {
        '0.0.1': 'Alpha release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(None, None, 3)),
            }),
            homepage='http://r0k.us/graphics/kodak/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        paths = dl_manager.download_and_extract(IMAGE_URLS)

        return {
            'test': self._generate_examples(paths),
        }

    def _generate_examples(self, paths):
        """Yields examples."""
        for key, file in paths.items():
            yield key, {
                'image': file,
            }
