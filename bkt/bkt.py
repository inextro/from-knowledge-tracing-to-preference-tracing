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
    top_weighted = args.top_weighted # 가장 비중이 높은 토픽만 MC로 고려하는 경우 True
    ratio = args.ratio
    forgets = args.forgets
    multigs = args.multigs

    if not top_weighted and ratio is None:
        raise ValueError('Flag가 True이면 반드시 ratio를 입력해야합니다.')

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

    print('start training')
    print(f'top weighted: {top_weighted}')
    print(f'forgets: {forgets}, multigs: {multigs}, num_topics: {num_topics}, ratio: {ratio}')
    print(f'avg mcs: {avg_mc}')
    model.fit(data=df, defaults=defaults, forgets=forgets, multigs=multigs)
    # model.save(f'bkt_forget={forgets}_gs={multigs}_Topic={num_topics}.pkl')
    print(model.evaluate(data=df, metric=metrics))