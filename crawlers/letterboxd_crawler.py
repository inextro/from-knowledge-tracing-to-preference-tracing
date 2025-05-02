import os
import time
import random
import requests
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import quote


def get_url(query, base_url='https://letterboxd.com/search/'):
    return base_url + quote(query)


# 줄거리가 크롤링되지 않은 영화들만 추출
movies = pd.read_csv('../data/crawled_data/movie_wiki.csv')

no_plots = movies[(movies['plots'].isna()) | (movies['plots'] == 'no plots')]
no_plots = no_plots.reset_index(drop=True)


# 해당 영화들의 줄거리를 letterboxd에서 크롤링
urls = []
plots = []
base = 'https://letterboxd.com/'

for i in tqdm(range(no_plots.shape[0])):
    # 첫번째 페이지
    response = requests.get(get_url(query=no_plots.loc[i, 'title']))
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')

    try:
        first_result = soup.find('span', {'class': 'film-title-wrapper'})
        new_url = base + first_result.find('a')['href']

    except AttributeError:
        urls.append('no results')
        plots.append('no results')
        continue

    else:
        # 두번째 페이지
        response = requests.get(new_url)
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')

        try:
            synopsis_part = soup.find('div', {'class': 'truncate'})
            synopsis = synopsis_part.text.strip()
            urls.append(new_url)
            plots.append(synopsis)

        except AttributeError:
            urls.append(new_url)
            plots.append('no synopsis')
    
    finally:
        time.sleep(random.uniform(0.3, 1))


# 크롤링한 정보들을 no_plots에 추가
no_plots['url'] = urls
no_plots['plots'] = plots

# 결과 저장
save_path = '../data/crawled_data'

if not os.path.exists(save_path):
    os.mkdir(save_path)

no_plots.to_csv(os.path.join(save_path, 'movie_lbox.csv'), index=False)