import os
import time
import random
import requests
import pandas as pd

from tqdm import tqdm
from bs4 import BeautifulSoup
from urllib.parse import quote
from urllib.parse import unquote
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_url(query):
    base_url = 'https://www.google.com/search?q='
    encoded_query = quote(query + ' wikipedia')

    return base_url + encoded_query


def start_brower():
    # 브라우저 초기화
    driver = webdriver.Chrome()

    return driver


def get_html_content(page_title):
    api_url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'parse', 
        'page': page_title, 
        'prop': 'text', 
        'format': 'json'
    }
    response = requests.get(api_url, params=params)
    response_json = response.json()

    return response_json['parse']['text']['*']


# 데이터 로드
# movies = pd.read_csv('data/ml-latest-small/movies.csv')
movies = pd.read_csv('../data/ml-1m/movies.dat', delimiter='::', engine='python', encoding='ISO-8859-1', names=['movieId', 'title', 'genres'])

# 중복되는 영화 제거
# movies = movies.drop(labels=[5601, 9468, 4169, 5854, 6932], axis=0)
# movies = movies.reset_index(drop=True)


# urls, plots 초기화
urls = []
plots = []

driver = webdriver.Chrome()

for i in tqdm(range(movies.shape[0])):
    if i % 300 == 0: # 300개 크롤링 이후 브라우저 재시작
        driver.quit()
        driver = start_brower()

    driver.get(get_url(query=movies['title'][i]))

    # 첫번째 검색결과의 url 저장
    first_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'h3.LC20lb.MBeuO.DKV0Md'))
    )
    link_element = first_element.find_element(By.XPATH, '..')
    url = link_element.get_attribute('href')
    urls.append(url)


    # BeautifulSoup로 html 줄거리만 추출하기
    try:
        page_title = url.split('/')[-1]
        page_title = unquote(page_title)
        html_content = get_html_content(page_title=page_title)

    except KeyError: 
        plots.append('no plots')
    
    else:
        soup = BeautifulSoup(html_content, 'html.parser')

        plot_section = None
        for section_title in ['Plot', 'Plot Summary', 'Summary', 'Synopsis']:
            plot_section = soup.find('span', id=section_title)

            if plot_section:
                break
        
        if plot_section:
            plot_text = ''

            for sibling in plot_section.parent.find_next_siblings():
                if sibling.name == 'h2':
                    break
                if sibling.name == 'p':
                    plot_text += sibling.text
            plots.append(plot_text.replace('\n', ' '))
        else:
            plots.append('no plots')

        time.sleep(random.uniform(0.3, 1))

driver.quit()


# 크롤링한 정보들을 movies에 추가
movies['url'] = urls
movies['plots'] = plots


# 결과 저장
save_path = '../data/crawled_data_1m'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# movies.to_csv('data/craweld_data/movies_wiki.csv', index=False)
movies.to_csv(os.path.join(save_path, 'movie_wiki.csv'), index=False)