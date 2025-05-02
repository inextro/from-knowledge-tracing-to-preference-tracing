import os
import re
import argparse
import requests
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import quote


def get_query_url(title, exact=False):
    base_url = 'https://www.imdb.com/find/?q='
    encoded_title = quote(title)
    exact_match = '&s=tt&exact=true'
    
    if exact: # exact_match option을 사용
        # 실제로 검색에 사용될 url
        query_url = base_url + encoded_title + exact_match
    else: # exact_match option을 사용하지 않음
        query_url = base_url + encoded_title

    return query_url


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exact', '-e', type=bool, default=False)
    args = parser.parse_args()


    # 영화 데이터베이스 불러오기
    movies = pd.read_csv('../data/ml-1m/movies.dat', delimiter='::', engine='python', encoding='ISO-8859-1', names=['movieId', 'title', 'genres'])


    # 헤더 정보; 헤더가 없으면 페이지에 접근이 불가능함
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }


    # imdb에서 영화들을 크롤링
    imdb_ids = []
    plots = []

    for i in tqdm(range(movies.shape[0])):
        query_url = get_query_url(title=movies.loc[i, 'title'], exact=False)

        response = requests.get(query_url, headers=headers)

        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')

        title_section = soup.find('section', {'data-testid': 'find-results-section-title', 'class': 'ipc-page-section ipc-page-section--base sc-17bafbdb-0 foOYdE'})
        
        try:
            first_result = title_section.find('ul').find('li')
            id_piece = first_result.find('a')['href']

            match = re.search(r'tt\d+', id_piece) # imdb id 추출
            imdb_id = match.group()

        except AttributeError: # 검색된 영화가 없다면
            imdb_ids.append('no result')
            plots.append('no result')
            continue # 해당 반복을 종료하고 다음 영화로 넘어가기

        base_url = 'https://www.imdb.com/title/'
        synopsis_url = '/plotsummary/?ref_=tt_stry_pl'

        response = requests.get(base_url + imdb_id + synopsis_url, headers=headers)

        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')

        try:
            synopsis_section = soup.find('div', {'data-testid': 'sub-section-synopsis'})
            synopsis = synopsis_section.find('div', class_='ipc-html-content-inner-div')
            imdb_ids.append(imdb_id)
            plots.append(synopsis.text)

        except AttributeError: # synopsis_section이 None인 경우 실행
            imdb_ids.append(imdb_id)
            plots.append('no synopsis')
    
    # 크롤링 결과물 저장
    movies['imdb_id'] = imdb_ids
    movies['plots'] = plots

    save_path = '../data/crawled_data'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if args.exact:
        movies.to_csv(os.path.join(save_path, 'movie_imdb_exact.csv'), index=False)
    else:
        movies.to_csv(os.path.join(save_path, 'movie_imdb.csv'), index=False)