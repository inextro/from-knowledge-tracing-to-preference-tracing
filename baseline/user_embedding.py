import numpy as np
import pandas as pd

from surprise import Dataset, Reader, SVD


def get_user_embedding(rating_data, emb_dim):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rating_data[['userId', 'movieId', 'rating']], reader)
    train_data = data.build_full_trainset()

    svd = SVD(n_factors=emb_dim)
    svd.fit(train_data)

    user_embeddings = {}

    for user_id in rating_data['userId'].unique():
        user_embeddings[user_id] = svd.pu[train_data.to_inner_uid(user_id)]

    df_user_embeddings = pd.DataFrame.from_dict(user_embeddings, orient='index')
    df_user_embeddings.columns = [f'embedding_{i}' for i in range(df_user_embeddings.shape[1])]

    return df_user_embeddings