import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, TfidfModel

df = pd.read_csv('sk_tokens.csv', header = 0)
# df = pd.read_csv(r'C:\Users\Public\Documents\sk_hynix\SKhynix_Project\sk_tokens.csv', header = 0)
df = df.dropna()
print(df.head())

# 문자열로 바뀐 'tokens'칼럼값을 리스트로 변환
df['tokens'] = df['tokens'].str.replace(r'[\[\'\]]', '', regex=True)
df['tokens'] = df['tokens'].str.split(', ')

## LDA
tokenized_docs = df['tokens']
id2word = corpora.Dictionary(tokenized_docs)
corpus_TDM = [id2word.doc2bow(doc) for doc in tokenized_docs]
tfidf = TfidfModel(corpus_TDM)
corpus_TFIDF = tfidf[corpus_TDM]

n = 50 # 토픽개수
lda = LdaModel(corpus=corpus_TFIDF,
               id2word=id2word,
               num_topics=n,
               random_state=100)

for t in lda.print_topics(num_topics=n):
  print(t)

