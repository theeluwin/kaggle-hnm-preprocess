import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.io import imsave
from skimage.transform import resize

from tools import utils


def get_resized_image(image: np.ndarray, max_size=256) -> np.ndarray:
    image_height, image_width, _ = image.shape
    original_size = max(image_height, image_width)
    resized_size = min(original_size, max_size)
    if image_width > image_height:
        new_width = int(min(resized_size, image_width))
        new_height = int(image_height * new_width / image_width)
    else:
        new_height = int(min(resized_size, image_height))
        new_width = int(image_width * new_height / image_height)
    resized_image = resize(image, (new_height, new_width), anti_aliasing=True, mode='reflect', preserve_range=True)
    resized_image = resized_image.astype(np.uint8)
    return resized_image


def entry():
    df_item = pd.read_parquet('data/df_item_preprocessed.pq')
    for row in tqdm(df_item.itertuples(), total=len(df_item)):
        ipath = row.ipath
        oipath = f'../raw/{ipath}'
        ripath = f'data/{ipath}'
        image = utils.imread_safe(oipath)
        resized = get_resized_image(image)
        os.makedirs(os.path.dirname(ripath), exist_ok=True)
        imsave(ripath, resized)


if __name__ == '__main__':
    entry()
