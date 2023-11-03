{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4752e35f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: gensim in c:\\users\\imray\\anaconda3\\lib\\site-packages (4.1.2)\n",
      "Requirement already satisfied: numpy>=1.17.0 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from gensim) (1.24.4)\n",
      "Requirement already satisfied: scipy>=0.18.1 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from gensim) (1.9.1)\n",
      "Requirement already satisfied: smart-open>=1.8.1 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from gensim) (5.2.1)\n"
     ]
    }
   ],
   "source": [
    "!pip install gensim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8814f626",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>morphs</th>\n",
       "      <th>pos</th>\n",
       "      <th>tokens</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2018-10-01 20:05:13</td>\n",
       "      <td>[한겨레] 4일 충북 청주에 M15 준공…낸드 주로 생산내년 5세대 낸드 생산해 중...</td>\n",
       "      <td>[ 한겨레 ] 4 일 충북 청주 에 M 15 준공 … 낸드 주로 생산 내년 5 세대...</td>\n",
       "      <td>[('[', 'SSO'), ('한겨레', 'NNG'), (']', 'SSC'), (...</td>\n",
       "      <td>['한겨레', '충북', '청주', '준공', '낸드', '낸드', '중국', '격...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2018-10-01 18:59:24</td>\n",
       "      <td>치킨게임서 승자만 살아남아\"하나멤버스·하이로보 서비스고객에 1대1 맞춤 제공 목표\"...</td>\n",
       "      <td>치킨 게 임서 승자 만 살아남 아 \" 하나 멤 버스 · 하이 로 보 서 비스 고객 ...</td>\n",
       "      <td>[('치킨', 'VV+ETM'), ('게', 'NNB+JKS'), ('임서', 'N...</td>\n",
       "      <td>['임서', '승자', '버스', '하이', '비스', '고객', '맞춤', '제공...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2018-10-01 18:58:49</td>\n",
       "      <td>ADT캡스 인수 박정호 SKT사장1인가구 + 토털케어서비스 집중드론보안 등 개척 서...</td>\n",
       "      <td>ADT 캡스 인수 박정호 SKT 사장 1 인가 구 + 토털 케어 서비스 집중 드론 ...</td>\n",
       "      <td>[('ADT', 'SL'), ('캡스', 'NNP'), ('인수', 'NNP'), ...</td>\n",
       "      <td>['캡스', '인수', '박정호', '토털', '케어', '서비스', '집중', '...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     1                                                  2  \\\n",
       "0  2018-10-01 20:05:13  [한겨레] 4일 충북 청주에 M15 준공…낸드 주로 생산내년 5세대 낸드 생산해 중...   \n",
       "1  2018-10-01 18:59:24  치킨게임서 승자만 살아남아\"하나멤버스·하이로보 서비스고객에 1대1 맞춤 제공 목표\"...   \n",
       "2  2018-10-01 18:58:49  ADT캡스 인수 박정호 SKT사장1인가구 + 토털케어서비스 집중드론보안 등 개척 서...   \n",
       "\n",
       "                                              morphs  \\\n",
       "0  [ 한겨레 ] 4 일 충북 청주 에 M 15 준공 … 낸드 주로 생산 내년 5 세대...   \n",
       "1  치킨 게 임서 승자 만 살아남 아 \" 하나 멤 버스 · 하이 로 보 서 비스 고객 ...   \n",
       "2  ADT 캡스 인수 박정호 SKT 사장 1 인가 구 + 토털 케어 서비스 집중 드론 ...   \n",
       "\n",
       "                                                 pos  \\\n",
       "0  [('[', 'SSO'), ('한겨레', 'NNG'), (']', 'SSC'), (...   \n",
       "1  [('치킨', 'VV+ETM'), ('게', 'NNB+JKS'), ('임서', 'N...   \n",
       "2  [('ADT', 'SL'), ('캡스', 'NNP'), ('인수', 'NNP'), ...   \n",
       "\n",
       "                                              tokens  \n",
       "0  ['한겨레', '충북', '청주', '준공', '낸드', '낸드', '중국', '격...  \n",
       "1  ['임서', '승자', '버스', '하이', '비스', '고객', '맞춤', '제공...  \n",
       "2  ['캡스', '인수', '박정호', '토털', '케어', '서비스', '집중', '...  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from gensim import corpora\n",
    "from gensim.models import LdaModel, TfidfModel\n",
    "\n",
    "df = pd.read_csv('stopWordedToken.csv')\n",
    "df.drop(['Unnamed: 0'], axis=1, inplace=True)\n",
    "df.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "59444c42",
   "metadata": {},
   "outputs": [],
   "source": [
    "from gensim import corpora\n",
    "from gensim.models import LdaModel, TfidfModel\n",
    "## LDA\n",
    "tokenized_docs = df['tokens'].dropna().apply(lambda x: ' '.join(x)).apply(lambda x: x.split())\n",
    "id2word = corpora.Dictionary(tokenized_docs)\n",
    "corpus_TDM = [id2word.doc2bow(doc) for doc in tokenized_docs]\n",
    "tfidf = TfidfModel(corpus_TDM)\n",
    "corpus_TFIDF = tfidf[corpus_TDM]\n",
    "\n",
    "n = 50 # 토픽개수\n",
    "lda = LdaModel(corpus=corpus_TFIDF,\n",
    "               id2word=id2word,\n",
    "               num_topics=n,\n",
    "               random_state=100)\n",
    "\n",
    "for t in lda.print_topics(num_topics=n):\n",
    "  # print(t)\n",
    "  pass\n",
    "\n",
    "# for i, row in df.iterrows():\n",
    "#   tokens = df['tokens'][i]\n",
    "\n",
    "#   topic_info_ = lda[corpus_TFIDF]\n",
    "#   dominant_topic = sorted(topic_info_, key=lambda x: x[1], reverse=True)[0]\n",
    "#   df.loc[i, 'topic'] = dominant_topic[0]\n",
    "#   df.loc[i, 'topic_details'] = dominant_topic[1]\n",
    "\n",
    "\n",
    "# df.to_csv('/content/drive/MyDrive/sesac_jongro/SK하이닉스/stopwordedToken/결과.csv', index=False)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "59093024",
   "metadata": {},
   "outputs": [],
   "source": [
    "# !pip install pyLDAvis==3.4.1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "64ac8eb8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pyLDAvis in c:\\users\\imray\\anaconda3\\lib\\site-packages (3.4.1)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pyLDAvis) (2.11.3)\n",
      "Collecting pandas>=2.0.0\n",
      "  Using cached pandas-2.1.2-cp39-cp39-win_amd64.whl (10.8 MB)\n",
      "Requirement already satisfied: scipy in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pyLDAvis) (1.9.1)\n",
      "Requirement already satisfied: numpy>=1.24.2 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pyLDAvis) (1.24.4)\n",
      "Requirement already satisfied: numexpr in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pyLDAvis) (2.8.3)\n",
      "Requirement already satisfied: gensim in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pyLDAvis) (4.1.2)\n",
      "Requirement already satisfied: joblib>=1.2.0 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pyLDAvis) (1.3.2)\n",
      "Requirement already satisfied: funcy in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pyLDAvis) (2.0)\n",
      "Requirement already satisfied: scikit-learn>=1.0.0 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pyLDAvis) (1.0.2)\n",
      "Requirement already satisfied: setuptools in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pyLDAvis) (63.4.1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2023.3)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pandas>=2.0.0->pyLDAvis) (2022.1)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from scikit-learn>=1.0.0->pyLDAvis) (2.2.0)\n",
      "Requirement already satisfied: smart-open>=1.8.1 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from gensim->pyLDAvis) (5.2.1)\n",
      "Requirement already satisfied: MarkupSafe>=0.23 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from jinja2->pyLDAvis) (2.0.1)\n",
      "Requirement already satisfied: packaging in c:\\users\\imray\\anaconda3\\lib\\site-packages (from numexpr->pyLDAvis) (21.3)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.2->pandas>=2.0.0->pyLDAvis) (1.16.0)\n",
      "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from packaging->numexpr->pyLDAvis) (3.0.9)\n",
      "Installing collected packages: pandas\n",
      "  Attempting uninstall: pandas\n",
      "    Found existing installation: pandas 1.5.1\n",
      "    Uninstalling pandas-1.5.1:\n",
      "      Successfully uninstalled pandas-1.5.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ERROR: Could not install packages due to an OSError: [WinError 5] 액세스가 거부되었습니다: 'C:\\\\Users\\\\imray\\\\anaconda3\\\\Lib\\\\site-packages\\\\~andas\\\\_libs\\\\algos.cp39-win_amd64.pyd'\n",
      "Consider using the `--user` option or check the permissions.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "!pip install pyLDAvis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "f4f53f28",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: loky in c:\\users\\imray\\anaconda3\\lib\\site-packages (3.4.1)\n",
      "Requirement already satisfied: cloudpickle in c:\\users\\imray\\anaconda3\\lib\\site-packages (from loky) (2.0.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install loky"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b0725752",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import time\n",
    "import traceback\n",
    "from loky import set_loky_pickler\n",
    "from loky import get_reusable_executor\n",
    "from loky import wrap_non_picklable_objects"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b5f64c13",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pandas in c:\\users\\imray\\anaconda3\\lib\\site-packages (2.1.2)\n",
      "Requirement already satisfied: numpy<2,>=1.22.4 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pandas) (1.24.4)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pandas) (2022.1)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pandas) (2.8.2)\n",
      "Requirement already satisfied: tzdata>=2022.1 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install pandas -U"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "98e0bfc2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting pandas==1.5.1\n",
      "  Using cached pandas-1.5.1-cp39-cp39-win_amd64.whl (10.9 MB)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pandas==1.5.1) (2022.1)\n",
      "Requirement already satisfied: numpy>=1.20.3 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pandas==1.5.1) (1.24.4)\n",
      "Requirement already satisfied: python-dateutil>=2.8.1 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from pandas==1.5.1) (2.8.2)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\imray\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.1->pandas==1.5.1) (1.16.0)\n",
      "Installing collected packages: pandas\n",
      "  Attempting uninstall: pandas\n",
      "    Found existing installation: pandas 2.1.2\n",
      "    Uninstalling pandas-2.1.2:\n",
      "      Successfully uninstalled pandas-2.1.2\n",
      "Successfully installed pandas-1.5.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
      "pyldavis 3.4.1 requires pandas>=2.0.0, but you have pandas 1.5.1 which is incompatible.\n"
     ]
    }
   ],
   "source": [
    "!pip install pandas==1.5.1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0aaf27cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "<link rel=\"stylesheet\" type=\"text/css\" href=\"https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v1.0.0.css\">\n",
       "\n",
       "\n",
       "<div id=\"ldavis_el1146821705252324801369088109\" style=\"background-color:white;\"></div>\n",
       "<script type=\"text/javascript\">\n",
       "\n",
       "\n",
       "function LDAvis_load_lib(url, callback){\n",
       "  var s = document.createElement('script');\n",
       "  s.src = url;\n",
       "  s.async = true;\n",
       "  s.onreadystatechange = s.onload = callback;\n",
       "  s.onerror = function(){console.warn(\"failed to load library \" + url);};\n",
       "  document.getElementsByTagName(\"head\")[0].appendChild(s);\n",
       "}\n",
       "\n",
       "if(typeof(LDAvis) !== \"undefined\"){\n",
       "   // already loaded: just create the visualization\n",
       "   !function(LDAvis){\n",
       "       new LDAvis(\"#\" + \"ldavis_el1146821705252324801369088109\", ldavis_el1146821705252324801369088109_data);\n",
       "   }(LDAvis);\n",
       "}else if(typeof define === \"function\" && define.amd){\n",
       "   // require.js is available: use it to load d3/LDAvis\n",
       "   require.config({paths: {d3: \"https://d3js.org/d3.v5\"}});\n",
       "   require([\"d3\"], function(d3){\n",
       "      window.d3 = d3;\n",
       "      LDAvis_load_lib(\"https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js\", function(){\n",
       "        new LDAvis(\"#\" + \"ldavis_el1146821705252324801369088109\", ldavis_el1146821705252324801369088109_data);\n",
       "      });\n",
       "    });\n",
       "}else{\n",
       "    // require.js not available: dynamically load d3 & LDAvis\n",
       "    LDAvis_load_lib(\"https://d3js.org/d3.v5.js\", function(){\n",
       "         LDAvis_load_lib(\"https://cdn.jsdelivr.net/gh/bmabey/pyLDAvis@3.4.0/pyLDAvis/js/ldavis.v3.0.0.js\", function(){\n",
       "                 new LDAvis(\"#\" + \"ldavis_el1146821705252324801369088109\", ldavis_el1146821705252324801369088109_data);\n",
       "            })\n",
       "         });\n",
       "}\n",
       "</script>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pyLDAvis.gensim\n",
    "\n",
    "pyLDAvis.enable_notebook()\n",
    "vis = pyLDAvis.gensim.prepare(lda, corpus_TDM, id2word)\n",
    "pyLDAvis.display(vis)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a059e25",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "523bef84",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54ca7471",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0bf80596",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "592aa0fa",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}