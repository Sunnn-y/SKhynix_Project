import pandas as pd
import MeCab
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
from multiprocess import Pool

df = pd.read_csv('sk_data_del.csv')

# 데이터프레임을 작은 블록으로 분할
block_size = 1000
data_blocks = [df[i:i + block_size] for i in range(0, len(df), block_size)]

# 불용어 및 토큰 리스트 정의
stop_word = ['이날', '보다', '거래', '만원', '포인트', '종가', '지수', '세대', '어치', '개인', '대하', '이어',
             '뉴시스', '이틀', '전일', '거대', '박수민', '울엄마', '스터리', '간벌', '헤럴드', '항목', '약지',
             '%↓', '-', "\'…", '지디', '투데이', '기자', '학생', '그림', '학과', '동년배', '%)', '%,', 'com',
             '세에', 'chosun', '조선일보', '에게선', '오토캠핑', '전월', '연합뉴스', '대다수', '대중', 'kr',
             '합니다', '보였', '십니까', '참기름', '까마득', '문순', '일지', '고려대', '오르내리', '조기', '간밤',
             '강세', '식목일', '데이터', '학년도', '코스닥', '코스피', '하락', '매수', '순매도', '거래일',
             '금리', '외국인', '마감', '상승', '종목', '기관', '분기', '기업', '차량', '데일리안', '만기',
             '미스트', '경제', '작성', '구성원', '가정용품', '경기도', '가격', '정오뉴스', '금요일', '앵커',
             '어제', '그룹', '내년', '시스', '단위', '기사', '고르기', '달러', '종류', '중구', '당사', '팟캐스트',
             '대학', '명동', '상위', '사장', '이사', '계감', '초당', '이익', '동기', '데일리', '휴가', '순위',
             '스포츠서울', '분야', '생산', '한경', '복장', '검색', '대차', '열흘', '야구', '반면', '구역', '지난해',
             '전년', '어르신', '온제', '근처', '등급', '진작', '무인', '기사문']

N_pos = ['NNG', 'NNP']

def preprocess(text):
    filtered_text = []
    for word, pos in text:
        if len(word) >= 2 and word not in stop_word and pos in N_pos:
            filtered_text.append((word, pos))
    return filtered_text

# MeCab 초기화
mecab = MeCab.Tagger()  # MeCab 초기화

# total 데이터프레임 정의
total = pd.DataFrame()

# 불용어 처리한 토큰 리스트 생성
def make_tokens(text):
    tokens = preprocess(text)
    return [word for word, _ in tokens]

# 토큰화 및 품사 부착
def get_tokens(block):
    block['morphs'] = block['contents'].apply(lambda text: mecab.parse(text).split("\n"))
    block['pos'] = block['morphs'].apply(lambda morphs: [(line.split("\t")[0], line.split("\t")[1].split(",")[0])
                                                  for line in morphs if "\t" in line])
    block['tokens'] = block['pos'].apply(make_tokens)
    return block  # 수정: 처리된 블록 반환

def main():
    num_cores = 10
    pool = Pool(num_cores)
    # 수정: 처리된 블록들을 모아서 total 데이터프레임에 연결
    total = pd.concat(pool.map(get_tokens, data_blocks), ignore_index=True)
    # 결과 데이터프레임 출력 또는 다른 작업 수행
    print(total.head())  # 예시로 상위 몇 개의 행을 출력
    total.to_csv('sk_output.csv', index=False)

if __name__ == "__main__":
    main()

