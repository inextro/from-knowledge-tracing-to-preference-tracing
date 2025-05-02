import nltk
import pandas as pd

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# # 최초 한 번만 실행 
# nltk.download('stopwords')
# nltk.download('punkt')

def preprocessing(data: pd.Series):
    """
    데이터 전처리 순서
    1. (소문자로 변경 +) 특수문자 제거 후 토큰화
    2. pos-tagging(형용사, 동사, 명사, 부사)
    3. lemmatization
    4. 불용어 제거
    """
    result_1 = data.apply(lambda x: nltk.regexp_tokenize(x.lower(), '[A-Za-z1-9]+'))
    print('토큰화 완료')
    
    result_2 = result_1.apply(lambda x: pos_tagging(x))
    print('품사 태깅 완료')
    
    result_3 = result_2.apply(lambda x: lemmatize(x))
    print('어간 추출 완료')
    
    result_4 = result_3.apply(lambda x: remove_stopwords(x))
    print('불용어 제거 완료')
    
    return result_4

def get_wordnet_pos(tag):
    # J: 형용사, V: 동사, N: 명사, R: 부사
    if tag.startswith('J'):
        return wordnet.ADJ
    
    elif tag.startswith('V'):
        return wordnet.VERB
    
    elif tag.startswith('N'):
        return wordnet.NOUN
    
    elif tag.startswith('R'):
        return wordnet.ADV
    
    else:
        return None

def pos_tagging(tokens):
    result = []
    tagged = pos_tag(tokens)
    
    for word, tag in tagged:
        if tag.startswith(('J', 'V', 'N', 'R')):
            result.append((word, tag))
            
    return result

def lemmatize(tagged):
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) 
                  if get_wordnet_pos(tag) else word 
                  for word, tag in tagged]
    
    return lemmatized

def remove_stopwords(tokens):
    result = []
    stop_words = set(stopwords.words('english'))
    
    for token in tokens:
        if len(token) > 1 and token not in stop_words: # 토큰의 길이가 1보다 크고 stopwords가 아닌 토큰만 추출
            result.append(token)
    
    return result


if __name__ == '__main__':
    movies = pd.read_csv('../data/crawled_data/plots_crawled.csv')

    # plot이 수집되지 못한 영화 제거하기
    # plots 컬럼이 no results, no synopsis, NaN이 아닌 영화들
    movies = movies[(movies['plots'] != 'no results') & (movies['plots'] != 'no synopsis') & (~movies['plots'].isna())]

    preprocessed_data = preprocessing(data=movies['plots'].apply(lambda x: str(x)))
    preprocessed_data.to_csv('./preprocessed_data.csv', index=False)