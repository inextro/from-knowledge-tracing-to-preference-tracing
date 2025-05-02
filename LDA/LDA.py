import os
import ast
import gensim
import argparse
import pandas as pd

from gensim import corpora
from datetime import datetime


def remove_stopwords(tokens, custom_stopwords):
    result = []
    
    # 사용자 지정 불용어 추가
    stop_words = set(custom_stopwords['stopwords'])

    for token in tokens:
        if token not in stop_words: # stopwords가 아닌 토큰만 추출
            result.append(token)

    return result


def LDA(preprocessed_data, num_topics, top_n):
    # 토큰화된 텍스트 데이터로부터 사전을 생성 -> 각 토큰마다 정수 ID를 할당
    # 전처리가 끝난 문서들로 corpus를 생성
    dictionary = corpora.Dictionary(preprocessed_data) # Dictionary(전처리가 끝난 문서: texts)
    # dictionary.filter_extremes(no_below=30) # no_below: 최소한 n개 이상 문서에 출현한 단어만 선정
    corpus = [dictionary.doc2bow(text) for text in preprocessed_data] # for text in 전처리가 끝난 문서: texts

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, 
                                            num_topics=num_topics, 
                                            id2word=dictionary, 
                                            alpha='auto', 
                                            eta='auto', 
                                            minimum_probability=0, 
                                            random_state=4
                                            )
    
    # 문서별 토픽 비중
    topic_per_document_list = []
    
    for doc_id, doc in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc)
        topic_per_document_list.append(doc_topics)
        
    # 문서별 토픽 비중을 DataFrame으로 변환
    topic_per_document = {f'Topic {i}': [] for i in range(num_topics)}
    
    for doc in topic_per_document_list:
        # 각 문서에 대해서 모든 토픽의 비중을 초기화
        topic_distribution = [0] * num_topics
        
        for topic_num, topic_weight in doc:
            topic_distribution[topic_num] = topic_weight
        
        for i, weight in enumerate(topic_distribution):
            topic_per_document[f'Topic {i}'].append(weight)
    
    # 토픽별 단어 비중을 DataFrame으로 변환
    topics = lda_model.show_topics(num_topics=num_topics, num_words=top_n, formatted=False) # 각 토픽을 대표하는 상위 30개 단어 (default: 30개)
    word_per_topic = {f'Topic {i}': None for i in range(num_topics)}
    
    for topic, pair in topics:
        word_list = [word for word, weight in pair]
        word_per_topic[f'Topic {topic}'] = word_list
    
    # 저장 경로 확인 및 생성
    save_path = './result/result_' + str(num_topics)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    topic_per_document_df = pd.DataFrame(topic_per_document)
    topic_per_document_df.to_csv(os.path.join(save_path, datetime.now().strftime('%Y%m%d') + '_Topic=' + str(num_topics) + '_TopicDist.csv'), index=False)

    word_per_topic_df = pd.DataFrame(word_per_topic)
    word_per_topic_df.to_csv(os.path.join(save_path, datetime.now().strftime('%Y%m%d') + '_Topic=' + str(num_topics) + '_WordDist.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # argument 추가하기
    parser.add_argument('-n', '--num_topics', type=int, required=True, help='LDA 모델에서 사용될 토픽의 개수를 입력하세요.')
    parser.add_argument('-t', '--top_n', type=int, default=30, help='각 토픽을 대표하는 단어를 얼마나 출력할 것인지 입력하세요.')

    # 입력받은 argument를 parsing
    args = parser.parse_args()
    num_topics = args.num_topics
    top_n = args.top_n

    preprocessed_data = pd.read_csv('./preprocessed_data.csv')

    # 사용자 지정 불용어 제거
    custom_stopwords = pd.read_csv('./stopwords.csv')

    # 문자열 리스트를 리스트로 변환: ast 라이브러리 사용
    preprocessed_data['plots'] = preprocessed_data['plots'].apply(lambda x: ast.literal_eval(x))

    # 사용자 정의 불용어를 제거한 파일을 preprocessed_lda.csv로 저장
    preprocessed_lda = preprocessed_data['plots'].apply(lambda x: remove_stopwords(tokens=x, custom_stopwords=custom_stopwords))
    preprocessed_lda.to_csv('./preprocessed_lda.csv', index=False)

    # LDA 코드 실행
    LDA(preprocessed_data=preprocessed_lda, num_topics=num_topics, top_n=top_n)