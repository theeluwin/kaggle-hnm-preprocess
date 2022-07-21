import pickle

import torch
import numpy as np
import pandas as pd

from PIL import Image

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from sklearn.decomposition import PCA

from tools.utils import DTimer
from tools.layers import get_backbone_by_name


class ExtractDataset(Dataset):

    def __init__(self):

        # init
        self.transform = transforms.Compose([
            transforms.Resize(240),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),
        ])

        # load data
        data = []
        df_item = pd.read_parquet('data/df_item_preprocessed.pq')
        for row in df_item.itertuples():
            iid = row.Index
            ipath = row.ipath
            path = f'rough/{ipath}'
            data.append((iid, path))
        self.data = data

    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        return self.transform(image)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        iid, path = self.data[index]
        return {
            'iid': iid,
            'image': self.load_image(path),
        }


def entry():

    # init
    with DTimer() as dtimer:
        model = get_backbone_by_name('EfficientNet-b1')
        model = model.eval()
        model = model.cuda()
        dataset = ExtractDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=32,
            pin_memory=True,
            drop_last=False
        )
    print(f"init done (elapsed: {dtimer.elapsed})")

    # extract
    with DTimer() as dtimer:
        iids = []
        imagevecs = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="extract image vectors"):
                images = batch['image'].cuda()
                vectors = model.extract_features(images)
                vectors = vectors.detach().cpu().numpy()
                iids.extend(batch['iid'])
                imagevecs.extend(list(vectors))
    print(f"extract done (elapsed: {dtimer.elapsed})")

    # intermission
    del dataloader
    del dataset
    del model

    # save raw (512)
    with DTimer() as dtimer:
        iid2imagevec = {}
        for iid, imagevec in zip(iids, imagevecs):
            iid2imagevec[iid] = imagevec
        with open('data/iid2imagevec.pkl', 'wb') as fp:
            pickle.dump(iid2imagevec, fp)
    print(f"raw save done (elapsed: {dtimer.elapsed})")

    # PCA
    with DTimer() as dtimer:
        X = np.array(imagevecs)
        pca = PCA(n_components=128)
        X = pca.fit_transform(X)
    print(f"PCA done (elapsed: {dtimer.elapsed})")

    # save PCA
    with DTimer() as dtimer:
        iid2imagevec = {}
        for iid, imagevec in zip(iids, X):
            iid2imagevec[iid] = imagevec
        with open('data/iid2imagevec_PCA.pkl', 'wb') as fp:
            pickle.dump(iid2imagevec, fp)
    print(f"PCA save done (elapsed: {dtimer.elapsed})")

    # intermission
    del iid2imagevec
    del imagevecs
    del pca
    del X


if __name__ == '__main__':
    entry()
