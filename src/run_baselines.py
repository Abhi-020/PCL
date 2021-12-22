#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB


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
    

# analysis
'''
cf = [i for i in range(50,1000,50)]
ac = []

for i in cf:
    cv = CountVectorizer( max_features = i)
    
    X = cv.fit_transform(corpus).toarray()
    y = df.iloc[:,-1].values
    
    X_train, X_test, y_train_task1, y_test_task1 = train_test_split(X, y, test_size = 0.20, random_state= 0)
    
    classifier = GaussianNB()
   # classifier = MultinomialNB()
    classifier.fit (X_train, y_train_task1)
    
    y_pred = classifier.predict(X_test)
    print(f'feature count {i}, {accuracy_score(y_test_task1, y_pred)}')
    ac.append([i,accuracy_score(y_test_task1, y_pred)])
'''   


# TF- IDF
# 

cv = TfidfVectorizer(min_df=1,stop_words='english')
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 0)
   

y_train_task1 = [ 0 if (y == 1 or y == 0) else 1 for y in y_train ]
y_test_task1 = [ 0 if (y == 1 or y == 0) else 1 for y in y_test ]


# In[89]:

#classifier = MultinomialNB()
classifier = GaussianNB() 
classifier.fit(X_train, y_train_task1)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test_task1, y_pred)
print(cm)


print(accuracy_score(y_test_task1, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test_task1, y_pred))


# In[39]:


#weighted array
#wt_array =len(X_train)/(len(set(y_train))*(np.bincount(y_train)))


# |Models|Description|Precion|Recall|weighted average F1|Accuracy F1|Remark|
# |---|---|---|---|---|---|---|
# |MultinomialNB|TFIDF Feature|.67|.82|.73|81.6|Highly biased|
# |GaussianNB|TFIDF Feature|.70|.69|.69|68.7|partial towards class 0|
# |GaussianNB|CV Feature|.75|.21|.30|21.4||
# |MultinomialNB|CV Feature|.74|.76|.75|76.2|1. overall accc,pre,re is stable throughout.2.class 2 doesn,t given any weightage|
# |GaussianNB|Gensim word2vec||||||

# WORD2VEC

'''model = Word2Vec(corpus, min_count = 1)

post_text = df.text.apply(gensim.utils.simple_preprocess)

model = gensim.models.Word2Vec(window = 10, min_count =2, workers=4)
model.build_vocab(post_text, progress_per = 1000)

model.train(post_text, total_examples = model.corpus_count, epochs = model.epochs)


model.wv.most_similar("good")
model.wv.similarity(w1 ="king", w2="women")


# model training

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 0)


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit (X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


print(accuracy_score(y_test, y_pred))
'''