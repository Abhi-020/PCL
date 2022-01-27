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



import time
import torch
import random
from torch import nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# In[15]:


voc_size = 5000

messages = X.copy()
messages['text'][1]
messages.reset_index(inplace= True)


import nltk
import re
from nltk.corpus import stopwords

#nltk.download('stopwords')

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


onehot_repr = [one_hot(words, voc_size)for words in corpus]
onehot_repr

sent_length = 60
embedded_docs= pad_sequences(onehot_repr, padding = 'pre', maxlen = sent_length)
print(embedded_docs)

X_final= np.array(embedded_docs)
y_final=np.array(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.20, random_state= 0)

print(X_train.shape)

##########################################################################################

traindataloader= DataLoader(timedata(traindf, 7), 32, shuffle=True, drop_last=True)
testdataloader= DataLoader(timedata(testdf, 7), 32, shuffle=False, drop_last=True)

###########################################################################################



# model

class BiLSTMNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=100, output_dim=1, n_layers=1, bidir=True, bs=32):
        super(BiLSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device('cpu')
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, bidirectional=bidir)
        self.d =  2 if bidir else 1
        self.fc = nn.Linear(hidden_dim*self.d, output_dim)
        self.relu = nn.ReLU()
        
        self.hidden = (torch.zeros(1*self.d,1*bs,self.hidden_dim),
                            torch.zeros(1*self.d,1*bs,self.hidden_dim))
        
    def forward(self, x):
        out, self.hidden = self.lstm(x, self.hidden)
        print(out.shape)
        out = self.fc(self.relu(out))
        return out



model = BiLSTMNet()



# results on test

y_pred = model1.predict(X_test)
classes_x=np.argmax(y_pred, axis =1)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


print(accuracy_score(y_test, classes_x))


from sklearn.metrics import classification_report
print(classification_report(y_test, classes_x))




