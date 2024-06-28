import argparse
import pandas as pd

from pyBKT.models import Model
from bkt_preprocessing import extract_mc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_topics', type=int, required=True)
    parser.add_argument('-t', '--top_weighted', action='store_true')
    parser.add_argument('-r', '--ratio', type=float, default=None)
    parser.add_argument('-f', '--forgets', action='store_true')
    parser.add_argument('-m', '--multigs', action='store_true')
    args = parser.parse_args()

    num_topics = args.num_topics
    top_weighted = args.top_weighted # 가장 비중이 큰 토픽을 MC로 정의 if True
    ratio = args.ratio
    forgets = args.forget
    multigs = args.multigs

    if not top_weighted and ratio is None:
        raise ValueError(f'You must specify the ratio if top_weighted is {top_weighted}.')

    defaults = {
        'order_id': 'timestamp', 
        'skill_name': 'mc', 
        'user_id': 'userId', 
        'correct': 'rating', 
        'multigs': 'movieId'
    }

    df = pd.read_csv('../data/ml-1m/ratings.csv')
    avg_mc, df = extract_mc(num_topics=num_topics, rating_data=df, flag=top_weighted, ratio=ratio)
    df = df.sort_values(by=['userId', 'timestamp'])
 
    model = Model(seed=52, num_fits=10)
    metrics = ['accuracy', 'auc', 'rmse']

    print('Start training')
    print(f'Top weighted: {top_weighted}')
    print(f'Forgets: {forgets}, Multigs: {multigs}, num_topics: {num_topics}')
    print(f'Avg mcs: {avg_mc}')
    model.fit(data=df, defaults=defaults, forgets=forgets, multigs=multigs)
    # model.save(f'bkt_forget={forgets}_gs={multigs}_Topic={num_topics}.pkl')
    print(model.evaluate(data=df, metric=metrics))