import pandas as pd
import MeCab
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel, TfidfModel
import pickle

df = pd.read_pickle(r'C:\\Users\\Hyoju\\Downloads\\sk_data_mecab_userdic.pkl')
# df = pd.read_csv('sk_data.csv')


## LDA
tokenized_docs = df['tokens'].dropna().apply(lambda x: ' '.join(x)).apply(lambda x: x.split())
id2word = corpora.Dictionary(tokenized_docs)
corpus_TDM = [id2word.doc2bow(doc) for doc in tokenized_docs]
tfidf = TfidfModel(corpus_TDM)
corpus_TFIDF = tfidf[corpus_TDM]

# corpus_TFIDF와 id2word를 pkl파일로 저장
with open('corpus_TFIDF.pkl', 'wb') as file:
    pickle.dump(corpus_TFIDF, file)

with open('id2word.pkl', 'wb') as file:
    pickle.dump(id2word, file)


n = 30 # 토픽개수
lda = LdaModel(corpus=corpus_TFIDF,
               id2word=id2word,
               num_topics=n,
               random_state=100)

# for t in lda.print_topics(num_topics=n):
#   print(t)