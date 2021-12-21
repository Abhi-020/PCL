#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.naive_bayes import MultinomialNB
#from gensim.models import Word2Vec


# In[2]:


df = pd.read_csv('../Data/dontpatronizeme_pcl.tsv', sep = '\t', names=['id','info','country', 'text','class'] )


# In[3]:


df.head()


# In[4]:


sns.set_style('whitegrid')
sns.countplot(x='class',data= df)


# In[5]:


sns.set_style('whitegrid')
sns.countplot(x='info',data= df, linewidth= 5)


# In[6]:


sns.set_style('whitegrid')
sns.countplot(x='class',hue='country',data=df,palette='RdBu_r')


# In[7]:


sns.set_style('whitegrid')
sns.countplot(x='class',hue='info',data=df,palette='rainbow')


# In[8]:


df['text'].count()


# In[9]:


df['info'].count()


# In[10]:


df = df.dropna(inplace = False)


# In[11]:


df = df.reset_index(drop = True)
df.info()


# In[12]:


import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


# In[13]:


corpus =[]
from tqdm import tqdm 
for i in tqdm(range(0,10468)):
    
    review = re.sub('[^a-zA-Z]',' ', df['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set (all_stopwords)]
    review =' '.join(review)
    corpus.append(review)
    


# In[14]:


corpus


# In[85]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 0)
    
   


# In[86]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[87]:


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
    


# TF- IDF
# 

# In[43]:


#cv = TfidfVectorizer(min_df=1,stop_words='english')
#X = cv.fit_transform(corpus).toarray()
#y = df.iloc[:,-1].values


# In[88]:


y_train_task1 = [ 0 if (y == 1 or y == 0) else 1 for y in y_train ]
y_test_task1 = [ 0 if (y == 1 or y == 0) else 1 for y in y_test ]


# In[89]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
#classifier = MultinomialNB()
classifier = GaussianNB() 
classifier.fit(X_train, y_train_task1)


# In[90]:


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test_task1, y_pred)
print(cm)


# In[91]:


accuracy_score(y_test_task1, y_pred)


# In[50]:


len(cv.get_feature_names())


# In[78]:


#X_train.inverse_transform(X_train[0])


# Dividing dataset into train and test
# 

# In[95]:


#np.array(X_train.iloc[0])


# In[59]:


len(X_train)


# In[72]:


len(y_train_task1)


# In[75]:


y_pred


# In[26]:


X_test[0]


# In[84]:


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

# In[20]:


import gensim
from gensim.models import Word2Vec


# In[21]:


model = Word2Vec(corpus, min_count = 1)


# In[22]:


post_text = df.text.apply(gensim.utils.simple_preprocess)
post_text


# In[23]:


model = gensim.models.Word2Vec(window = 10, min_count =2, workers=4)


# In[24]:


model.build_vocab(post_text, progress_per = 1000)


# In[25]:


model.epochs


# In[26]:


model.corpus_count


# In[27]:


model.train(post_text, total_examples = model.corpus_count, epochs = model.epochs)


# In[28]:


model.wv.most_similar("good")


# In[29]:


model.wv.similarity(w1 ="king", w2="women")


# In[ ]:





# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 0)


# In[48]:


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit (X_train, y_train)


# In[49]:


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[46]:


accuracy_score(y_test, y_pred)


# In[ ]:




