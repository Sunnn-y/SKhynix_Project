import pandas as pd
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
import pickle


# corpus_TFIDF 로드
with open('corpus_TFIDF.pkl', 'rb') as file:
    corpus_TFIDF = pickle.load(file)

# id2word 로드
with open('id2word.pkl', 'rb') as file:
    id2word = pickle.load(file)


n = 30 # 토픽개수
lda = LdaModel(corpus=corpus_TFIDF,
               id2word=id2word,
               num_topics=n,
               random_state=100)

# for t in lda.print_topics(num_topics=n):
#   print(t)