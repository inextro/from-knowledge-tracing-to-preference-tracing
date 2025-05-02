import argparse
import numpy as np
import pandas as pd


def extract_mc(num_topics, rating_data, flag=False, ratio=None):
    movies = pd.read_csv('../data/crawled_data/plots_crawled.csv')
    lda = pd.read_csv(f'../LDA/result/result_{num_topics}/20240626_Topic={num_topics}_TopicDist.csv')
    
    # 가장 비중이 큰 토픽을 mc로 가정하는 경우
    if flag:
        lda['mc'] = lda.apply(lambda x: np.argmax(x.values), axis=1)
        avg_mcs = 1
    # 일정 비중 이상의 토픽을 mc로 가정하는 경우
    else:
        lda['mc'] = lda.apply(lambda x: extract_mc_by_ratio(row=x, ratio=ratio), axis=1)
        lda['n_mc'] = lda['mc'].apply(lambda x: len(x))
        avg_mcs = lda['n_mc'].mean()
        
    movies = movies[(movies['plots'] != 'no results') & (movies['plots'] != 'no synopsis') & (~movies['plots'].isna())]
    lda['movieId'] = movies['movieId']

    assert(movies.shape[0] == lda.shape[0])

    rating_data = rating_data.merge(
        right=lda.loc[:, ['movieId', 'mc']], 
        how='inner', 
        on='movieId'
    )
    
    if not flag:
        rating_data = rating_data.explode(column='mc', ignore_index=True)

    return round(avg_mcs, 4), rating_data


def extract_mc_by_ratio(row, ratio):
    idx = row[row >= ratio].index
    mcs = [i.split()[1] for i in idx]

    return mcs