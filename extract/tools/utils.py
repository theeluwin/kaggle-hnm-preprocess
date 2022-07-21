import numpy as np

from typing import Tuple
from datetime import datetime as dt

from PIL import Image
from skimage.io import imread
from quadratum.functional import whiten


__all__: Tuple[str, ...] = (
    'DTimer',
    'ImageException',
    'imread_safe',
)


class DTimer(object):
    """ 사용 예시:

        with DTimer() as dtimer:
            for i in range(10):
                time.sleep(1)
        print(dtimer.elapsed)
    """
    def __enter__(self):
        self.start = dt.now()
        return self

    def __exit__(self, *args):
        self.end = dt.now()
        self.elapsed = self.end - self.start


class ImageException(Exception):
    pass


def imread_safe(path: str,
                min_size: int = 28
                ) -> np.ndarray:

    # read image
    try:
        image: np.ndarray = imread(path)
    except FileNotFoundError:
        raise ImageException("No such file.")
    except ValueError:
        raise ImageException("Not a valid image.")
    except OSError:
        raise ImageException("Not a valid file.")

    # get shape
    h: int
    w: int
    c: int
    try:
        h, w, c = image.shape
    except ValueError:
        image = np.stack((image,) * 3, axis=-1)
        try:
            h, w, c = image.shape
        except ValueError:
            raise ImageException(f"Improper shape: it should be 2 or 3 but got {len(image.shape)}.")

    # check shape
    if c < 3:
        raise ImageException("Only RGB is used.")
    if c > 3:
        image = whiten(image)
    if h < min_size or w < min_size:
        raise ImageException(f"Too small size: height - {h}, width - {w}.")

    # check variance
    if image.var() < 1.0:
        raise ImageException(f"Too small variance: {image.var()}.")

    # check PIL
    try:
        _ = Image.open(path).convert('RGB')
    except Exception:
        raise ImageException("PIL Error.")

    return image
