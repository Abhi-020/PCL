import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


import gensim
from gensim.models import Word2Vec

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


df_test = pd.read_csv('../Data/pcl_test.tsv', sep = '\t',names=['id','info','country', 'text'] )
df = pd.read_csv('../Data/dontpatronizeme_pcl.tsv', sep = '\t', names=['id','info','country', 'text','class'] )
df = df.dropna(inplace = False)

df = df.reset_index(drop = True)
df.info()

#nltk.download('stopwords')

ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

corpus =[]
from tqdm import tqdm 
for i in tqdm(range(0,10468)):
    
    review = re.sub('[^a-zA-Z]',' ', df['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set (all_stopwords)]
    review =' '.join(review)
    corpus.append(review)
for i in tqdm(range(0,3832)):    
    review = re.sub('[^a-zA-Z]',' ', df_test['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set (all_stopwords)]
    review =' '.join(review)
    corpus.append(review)   
    
# analysis
# TF- IDF CV

#cv = CountVectorizer( max_features = 1000)
cv = TfidfVectorizer(min_df=1,stop_words='english')
X = cv.fit_transform(corpus).toarray()
x_test= cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].values

X_train, X_val, y_train, y_val = train_test_split(X[:10468], y, test_size = 0.20, random_state= 0)
   

y_train_task1 = [ 0 if (y == 1 or y == 0) else 1 for y in y_train ]
y_val_task1 = [ 0 if (y == 1 or y == 0) else 1 for y in y_val ]

#logmodel = LogisticRegression(max_iter=1000)
logmodel = LogisticRegression(max_iter=1000, class_weight='balanced')
logmodel.fit(X_train, y_train_task1)
predictions = logmodel.predict(X_test)

accuracy = confusion_matrix(y_val_task1, predictions)
print(accuracy)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val_task1,predictions)
print(accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_val_task1, predictions))

y_pred = logmodel.predict(x_test)

S= open("Logistic_TASK_1", "w")
for i in y_pred:
    print(*i,sep=',', file= S)
S.close()

