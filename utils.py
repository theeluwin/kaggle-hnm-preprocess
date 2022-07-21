import os

import pandas as pd

from typing import List


def load_raw(raw_root: str):

    # load user
    if os.path.isfile(os.path.join(raw_root, 'customers.pq')):
        df_user_raw = pd.read_parquet(os.path.join(raw_root, 'customers.pq'))
    else:
        df_user_raw = pd.read_csv(os.path.join(raw_root, 'customers.csv'))
        df_user_raw.to_parquet(os.path.join(raw_root, 'customers.pq'))

    # load item
    if os.path.isfile(os.path.join(raw_root, 'articles.pq')):
        df_item_raw = pd.read_parquet(os.path.join(raw_root, 'articles.pq'))
    else:
        df_item_raw = pd.read_csv(os.path.join(raw_root, 'articles.csv'), dtype={'article_id': str})
        df_item_raw.to_parquet(os.path.join(raw_root, 'articles.pq'))

    # load log
    if os.path.isfile(os.path.join(raw_root, 'transactions_train.pq')):
        df_log_raw = pd.read_parquet(os.path.join(raw_root, 'transactions_train.pq'))
    else:
        df_log_raw = pd.read_csv(os.path.join(raw_root, 'transactions_train.csv'), dtype={'article_id': str})
        df_log_raw.to_parquet(os.path.join(raw_root, 'transactions_train.pq'))

    # load sub
    if os.path.isfile(os.path.join(raw_root, 'sample_submission.pq')):
        df_sub_raw = pd.read_parquet(os.path.join(raw_root, 'sample_submission.pq'))
    else:
        df_sub_raw = pd.read_csv(os.path.join(raw_root, 'sample_submission.csv'))
        df_sub_raw.to_parquet(os.path.join(raw_root, 'sample_submission.pq'))

    return (
        df_user_raw,
        df_item_raw,
        df_log_raw,
        df_sub_raw,
    )


def get_df_log_of(df_log, dname):
    df_log = df_log.copy()
    if dname == 'CV':
        df_log = df_log[(df_log['t_dat'] < '2019-10-02')]
        df_log['target'] = 'train'
        df_log.loc[(df_log['t_dat'] >= '2019-09-18'), 'target'] = 'valid'
        df_log.loc[(df_log['t_dat'] >= '2019-09-25'), 'target'] = 'test'
    elif dname == 'LB':
        df_log = df_log[(df_log['t_dat'] >= '2019-09-19')]
        df_log['target'] = 'train'
        df_log.loc[(df_log['t_dat'] >= '2020-09-16'), 'target'] = 'valid'
    else:
        raise Exception("dname = CV or LB only")
    return df_log


def calc_ap(answerset: set, predictions: List[str], top_k: int = 12) -> float:
    if not answerset:
        return 0.0
    if len(predictions) > top_k:
        predictions = predictions[:top_k]
    score = 0.0
    hit_count = 0.0
    seenset = set()
    for index, prediction in enumerate(predictions):
        if prediction in answerset and prediction not in seenset:
            hit_count += 1.0
            score += hit_count / (index + 1.0)
            seenset.add(prediction)
    return score / min(len(answerset), top_k)
