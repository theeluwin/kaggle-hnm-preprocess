import pickle

import numpy as np

from tqdm import tqdm

from tools.utils import DTimer


def entry():

    # load features
    with DTimer() as dtimer:
        with open('../data/iid2feature_noagg_CV.pkl', 'rb') as fp:
            iid2feature_CV = pickle.load(fp)
        with open('../data/iid2feature_noagg_LB.pkl', 'rb') as fp:
            iid2feature_LB = pickle.load(fp)
    print(f"feature loaded (elapsed: {dtimer.elapsed})")

    # load vectors raw
    with DTimer() as dtimer:
        with open('data/iid2imagevec.pkl', 'rb') as fp:
            iid2imagevec = pickle.load(fp)
        with open('data/iid2titlevec.pkl', 'rb') as fp:
            iid2titlevec = pickle.load(fp)
        with open('data/iid2descvec.pkl', 'rb') as fp:
            iid2descvec = pickle.load(fp)
    print(f"vector loaded (elapsed: {dtimer.elapsed})")

    # concat CV raw
    with DTimer() as dtimer:
        iid2catted_CV = {}
        for iid, feature in tqdm(iid2feature_CV.items(), "CV"):
            imagevec = iid2imagevec[iid]
            titlevec = iid2titlevec[iid]
            descvec = iid2descvec[iid]
            catted = np.concatenate((feature, imagevec, titlevec, descvec))
            iid2catted_CV[iid] = catted
        with open('data/iid2catted_CV.pkl', 'wb') as fp:
            pickle.dump(iid2catted_CV, fp)
        del iid2catted_CV
    print(f"concat CV raw done (elapsed: {dtimer.elapsed})")

    # concat LB raw
    with DTimer() as dtimer:
        iid2catted_LB = {}
        for iid, feature in tqdm(iid2feature_LB.items(), "LB"):
            imagevec = iid2imagevec[iid]
            titlevec = iid2titlevec[iid]
            descvec = iid2descvec[iid]
            catted = np.concatenate((feature, imagevec, titlevec, descvec))
            iid2catted_LB[iid] = catted
        with open('data/iid2catted_LB.pkl', 'wb') as fp:
            pickle.dump(iid2catted_LB, fp)
        del iid2catted_LB
    print(f"concat LB raw done (elapsed: {dtimer.elapsed})")

    # intermission
    del iid2imagevec
    del iid2titlevec
    del iid2descvec

    # load vectors PCA
    with DTimer() as dtimer:
        with open('data/iid2imagevec_PCA.pkl', 'rb') as fp:
            iid2imagevec_PCA = pickle.load(fp)
        with open('data/iid2titlevec_PCA.pkl', 'rb') as fp:
            iid2titlevec_PCA = pickle.load(fp)
        with open('data/iid2descvec_PCA.pkl', 'rb') as fp:
            iid2descvec_PCA = pickle.load(fp)
    print(f"vector loaded (elapsed: {dtimer.elapsed})")

    # concat CV PCA
    with DTimer() as dtimer:
        iid2catted_CV_PCA = {}
        for iid, feature in tqdm(iid2feature_CV.items(), "CV PCA"):
            imagevec = iid2imagevec_PCA[iid]
            titlevec = iid2titlevec_PCA[iid]
            descvec = iid2descvec_PCA[iid]
            catted = np.concatenate((feature, imagevec, titlevec, descvec))
            iid2catted_CV_PCA[iid] = catted
        with open('data/iid2catted_CV_PCA.pkl', 'wb') as fp:
            pickle.dump(iid2catted_CV_PCA, fp)
        del iid2catted_CV_PCA
    print(f"concat CV PCA done (elapsed: {dtimer.elapsed})")

    # concat LB
    with DTimer() as dtimer:
        iid2catted_LB_PCA = {}
        for iid, feature in tqdm(iid2feature_LB.items(), "LB"):
            imagevec = iid2imagevec_PCA[iid]
            titlevec = iid2titlevec_PCA[iid]
            descvec = iid2descvec_PCA[iid]
            catted = np.concatenate((feature, imagevec, titlevec, descvec))
            iid2catted_LB_PCA[iid] = catted
        with open('data/iid2catted_LB_PCA.pkl', 'wb') as fp:
            pickle.dump(iid2catted_LB_PCA, fp)
        del iid2catted_LB_PCA
    print(f"concat LB PCA done (elapsed: {dtimer.elapsed})")

    # intermission
    del iid2imagevec_PCA
    del iid2titlevec_PCA
    del iid2descvec_PCA


if __name__ == '__main__':
    entry()
