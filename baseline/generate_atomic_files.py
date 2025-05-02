import pandas as pd

# ml-1m: ratings.dat
ratings = pd.read_csv('./data/ml-1m/ratings.csv')
ratings.columns = ['user_id:token', 'item_id:token', 'label:float', 'timestamp:float']

ratings.to_csv('./dataset/ml/ml.inter', index=False)


# ml-1m: movies.dat
movies = pd.read_csv('./data/ml-1m/movies.dat', 
                     delimiter='::', 
                     engine='python', 
                     encoding='ISO-8859-1', 
                     names=['movieId', 'title', 'genres'])

movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies['movie_title'] = movies['title'].str.replace(r'\(\d{4}\)', "").str.strip()
movies['class'] = movies['genres'].str.replace('|', ' ')

movies = movies[['movieId', 'movie_title', 'release_year', 'class']]
movies.columns = ['item_id:token', 'movie_title:token_seq', 'release_year:token', 'class_token_seq']

movies.to_csv('./dataset/ml/ml.item', index=False)


# ml-1m: users.dat
users = pd.read_csv('./data/ml-1m/users.dat', 
                    delimiter='::', 
                    engine='python', 
                    encoding='ISO-8859-1', 
                    names=['userId', 'gender', 'age', 'occupation', 'zip-code'])

users = users[['userId', 'age', 'gender', 'occupation']]
users.columns = ['user_id:token', 'age:token', 'gender:token', 'occupation:token']

users.to_csv('./dataset/ml/ml.user')