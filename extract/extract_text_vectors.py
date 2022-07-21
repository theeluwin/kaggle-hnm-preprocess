import pickle

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import (
    Dataset,
    DataLoader,
)
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from tools.utils import DTimer


class ExtractDataset(Dataset):

    def __init__(self):

        # load data
        data = []
        df_item = pd.read_parquet('data/df_item_preprocessed.pq')
        for row in df_item.itertuples():
            iid = row.Index
            title = row.prod_name
            desc = row.detail_desc
            data.append((iid, title, desc))
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        iid, title, desc = self.data[index]
        return {
            'iid': iid,
            'title': title,
            'desc': desc,
        }


def entry():

    # init
    with DTimer() as dtimer:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        model = model.eval()
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
        titlevecs = []
        descvecs = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="extract text vectors"):
                titlevec = model.encode(batch['title'])
                descvec = model.encode(batch['desc'])
                iids.extend(batch['iid'])
                titlevecs.extend(list(titlevec))
                descvecs.extend(list(descvec))
    print(f"extract done (elapsed: {dtimer.elapsed})")

    # intermission
    del dataloader
    del dataset
    del model

    # title raw save (384)
    with DTimer() as dtimer:
        iid2titlevec = {}
        for iid, titlevec in zip(iids, titlevecs):
            iid2titlevec[iid] = titlevec
        with open('data/iid2titlevec.pkl', 'wb') as fp:
            pickle.dump(iid2titlevec, fp)
    print(f"title raw save done (elapsed: {dtimer.elapsed})")

    # title PCA
    with DTimer() as dtimer:
        X = np.array(titlevecs)
        pca = PCA(n_components=32)
        X = pca.fit_transform(X)
    print(f"title PCA done (elapsed: {dtimer.elapsed})")

    # title PCA save
    with DTimer() as dtimer:
        iid2titlevec = {}
        for iid, titlevec in zip(iids, X):
            iid2titlevec[iid] = titlevec
        with open('data/iid2titlevec_PCA.pkl', 'wb') as fp:
            pickle.dump(iid2titlevec, fp)
    print(f"title PCA save done (elapsed: {dtimer.elapsed})")

    # intermission
    del iid2titlevec
    del titlevecs
    del pca
    del X

    # desc raw save (384)
    with DTimer() as dtimer:
        iid2descvec = {}
        for iid, descvec in zip(iids, descvecs):
            iid2descvec[iid] = descvec
        with open('data/iid2descvec.pkl', 'wb') as fp:
            pickle.dump(iid2descvec, fp)
    print(f"desc raw save done (elapsed: {dtimer.elapsed})")

    # desc PCA
    with DTimer() as dtimer:
        X = np.array(descvecs)
        pca = PCA(n_components=128)
        X = pca.fit_transform(X)
    print(f"desc PCA done (elapsed: {dtimer.elapsed})")

    # desc PCA save
    with DTimer() as dtimer:
        iid2descvec = {}
        for iid, descvec in zip(iids, X):
            iid2descvec[iid] = descvec
        with open('data/iid2descvec_PCA.pkl', 'wb') as fp:
            pickle.dump(iid2descvec, fp)
    print(f"desc PCA save done (elapsed: {dtimer.elapsed})")

    # intermission
    del iid2descvec
    del descvecs
    del pca
    del X


if __name__ == '__main__':
    entry()
