import os
import argparse
import warnings
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from surprise import Dataset, Reader, SVD
from user_embedding import get_user_embedding
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


def rf_performance(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=4)
    rf.fit(X_train, y_train)
    
    acc = accuracy_score(y_true=y_test, y_pred=rf.predict(X_test))
    auc = roc_auc_score(y_true=y_test, y_score=rf.predict_proba(X_test)[:, 1])

    return round(acc, 4), round(auc, 4)


def kn_performance(X_train, X_test, y_train, y_test):
    kn = KNeighborsClassifier()
    kn.fit(X_train, y_train)

    acc = accuracy_score(y_true=y_test, y_pred=kn.predict(X_test))
    auc = roc_auc_score(y_true=y_test, y_score=kn.predict_proba(X_test)[:, 1])

    return round(acc, 4), round(auc, 4)


def lgbm_performance(X_train, X_test, y_train, y_test):
    lgbm = LGBMClassifier(random_state=4)
    lgbm.fit(X_train, y_train)

    acc = accuracy_score(y_true=y_test, y_pred=lgbm.predict(X_test))
    auc = roc_auc_score(y_true=y_test, y_score=lgbm.predict_proba(X_test)[:, 1])

    return round(acc, 4), round(auc, 4)


def xgb_performance(X_train, X_test, y_train, y_test):
    xgb = XGBClassifier(random_state=4)
    xgb.fit(X_train, y_train)

    acc = accuracy_score(y_true=y_test, y_pred=xgb.predict(X_test))
    auc = roc_auc_score(y_true=y_test, y_score=xgb.predict_proba(X_test)[:, 1])

    return round(acc, 4), round(auc, 4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_topics', type=int, required=True)
    parser.add_argument('-e', '--emb_dim', type=int, default=64)

    args = parser.parse_args()
    num_topics = args.num_topics
    emb_dim = args.emb_dim


    # load data
    movies = pd.read_csv('../data/crawled_data/plots_crawled.csv')
    ratings = pd.read_csv('../data/ml-1m/ratings.csv')
    # lda = pd.read_csv('../LDA/result/result_{0}/20240427_Topic={0}_TopicDist.csv'.format(num_topics))
    lda = pd.read_csv('../LDA/result/result_{0}/20240626_Topic={0}_TopicDist.csv'.format(num_topics))


    # preprocessing
    # movies = movies[(movies['plots'] != 'no results') & (movies['plots'] != 'no synopsis') & (~movies['plots'].isna())]
    lda['movieId'] = movies['movieId']
    merged_df = ratings.merge(right=lda, how='inner', on='movieId')
    topic_columns = ['Topic {}'.format(i) for i in range(num_topics)]
    df = merged_df[['userId', 'rating'] + topic_columns]
    df['rating'] = df['rating'].astype(float)


    # get user embedding
    df_user_embeddings = get_user_embedding(rating_data=ratings, emb_dim=emb_dim).reset_index(names='userId')
    df = df.merge(right=df_user_embeddings, how='inner', on='userId')


    # train-test split
    random_seed = 42
    np.random.seed(random_seed)

    train_ratio = 0.8
    train_count = int(df['userId'].nunique() * train_ratio)

    train_ids = np.random.choice(np.arange(1, train_count+1), size=train_count, replace=False)

    train_data = df[df['userId'].isin(train_ids)].reset_index(drop=True)
    test_data = df[~df['userId'].isin(train_ids)].reset_index(drop=True)

    X_train, y_train = train_data[[f'Topic {i}' for i in range(num_topics)] + [f'embedding_{j}' for j in range(emb_dim)]], train_data['rating'].values
    X_test, y_test = test_data[[f'Topic {i}' for i in range(num_topics)] + [f'embedding_{j}' for j in range(emb_dim)]], test_data['rating'].values

    
    # get performance
    print('rf start')
    rf_acc, rf_auc = rf_performance(X_train, X_test, y_train, y_test)
    print('rf end')

    print('knn start')
    kn_acc, kn_auc = kn_performance(X_train, X_test, y_train, y_test)
    print('knn end')


    print('lgbm start')
    lgbm_acc, lgbm_auc = lgbm_performance(X_train, X_test, y_train, y_test)
    print('lgbm end')

    print('xgb start')
    xgb_acc, xgb_auc = xgb_performance(X_train, X_test, y_train, y_test)
    print('xgb end')


    # save
    path = './result'
    
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, f'baseline_performance_Topic={num_topics}_Dim={emb_dim}.txt'), 'w') as f:
        f.write('model name: (acc, auc)\n')
        f.write(f'rf: {rf_acc, rf_auc}\n')
        f.write(f'kn: {kn_acc, kn_auc}\n')
        f.write(f'lgbm: {lgbm_acc, lgbm_auc}\n')
        f.write(f'xgb: {xgb_acc, xgb_auc}\n')