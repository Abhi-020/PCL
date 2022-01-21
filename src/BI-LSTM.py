#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np



df = pd.read_csv('../Data/dontpatronizeme_pcl.tsv', sep = '\t', names=['id','info','country', 'text','label'] )

df = df.dropna(inplace = False)

df = df.reset_index(drop = True)
df.info()

df_final = df[['text','label']]
df_final
df_final['label_']= [ 0 if (y == 1 or y == 0) else 1 for y in df_final['label']]
df_final
df_final.drop('label', axis = 1, inplace= True)
df_final.head()



X = df_final.drop('label_', axis =1) #getting independent variable
y = df_final['label_']



import tensorflow as tf

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional


# In[15]:


voc_size = 5000


# In[16]:


messages = X.copy()


# In[18]:


messages['text'][1]


# In[19]:


messages.reset_index(inplace= True)


# In[20]:


import nltk
import re
from nltk.corpus import stopwords


# In[21]:


nltk.download('stopwords')


# In[23]:


from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus =[]
from tqdm import tqdm 
for i in tqdm(range(0,10468)):
    
    review = re.sub('[^a-zA-Z]',' ', messages['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words ('english')]
    review =' '.join(review)
    corpus.append(review)
    


# In[25]:


corpus


# In[24]:


onehot_repr = [one_hot(words, voc_size)for words in corpus]
onehot_repr


# In[33]:


sent_length = 60
embedded_docs= pad_sequences(onehot_repr, padding = 'pre', maxlen = sent_length)
print(embedded_docs)


# In[34]:


embedded_docs[0]


# In[35]:


embedding_vector_features = 50
model =Sequential()
model.add(Embedding(voc_size,embedding_vector_features, input_length = sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation ='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer ='adam', metrics=['accuracy'])
print(model.summary())


# In[36]:


embedding_vector_features = 50
model1 =Sequential()
model1.add(Embedding(voc_size,embedding_vector_features, input_length = sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dense(1,activation ='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer ='adam', metrics=['accuracy'])
print(model1.summary())


# In[37]:


len(embedded_docs), y.shape


# In[38]:


X_final= np.array(embedded_docs)
y_final=np.array(y)


# In[39]:


X_final.shape, y_final.shape


# In[42]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.20, random_state= 0)


# In[62]:


y_train.shape, y_test.shape


# In[43]:


model.fit(X_train, y_train, validation_data =(X_test, y_test), epochs=10, batch_size =64)


# In[44]:


model1.fit(X_train, y_train, validation_data =(X_test, y_test), epochs=10, batch_size =64)


# In[70]:



y_pred = model1.predict(X_test)
classes_x=np.argmax(y_pred, axis =1)


# In[68]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[69]:


accuracy_score(y_test, classes_x)


# In[66]:


from sklearn.metrics import classification_report
print(classification_report(y_test, classes_x))


# In[ ]:




