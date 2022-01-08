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

#from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import GaussianNB


df = pd.read_csv('../Data/dontpatronizeme_pcl.tsv', sep = '\t', names=['id','info','country', 'text','class'] )
df = df.dropna(inplace = False)

df = df.reset_index(drop = True)
df.info()

#nltk.download('stopwords')

ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

'''corpus =[]
from tqdm import tqdm 
for i in tqdm(range(0,10468)):
    
    review = re.sub('[^a-zA-Z]',' ', df['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set (all_stopwords)]
    review =' '.join(review)
    corpus.append(review)'''
   
# analysis
post_text = (df.text.apply(gensim.utils.simple_preprocess))

model = gensim.models.Word2Vec(window = 10, min_count =3, workers=4)
model.build_vocab(post_text, progress_per = 500)
model.train(post_text, total_examples = model.corpus_count, epochs = model.epochs)
print(model.wv.similarity(w1="king", w2 = "queen"))

post_text = [] 
from tqdm import tqdm 
for i in tqdm(range(0,10468)):
    
    review = re.sub('[^a-zA-Z]',' ', df['text'][i])
    review = review.lower()
    review = review.split()
    review =' '.join(review)
    post_text.append(review)

# TF- IDF CV
# 
#cv = CountVectorizer( max_features = 500)
cv = TfidfVectorizer(min_df=1,stop_words='english')
X = cv.fit_transform(post_text).toarray()
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 0)
   

y_train_task1 = [ 0 if (y == 1 or y == 0) else 1 for y in y_train ]
y_test_task1 = [ 0 if (y == 1 or y == 0) else 1 for y in y_test ]

#logmodel = LogisticRegression(max_iter=1000)
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

clf = make_pipeline(StandardScaler(),
SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X_train, y_train_task1)

print("model fit")

predictions = clf.predict(X_test)
print(predictions)


accuracy = confusion_matrix(y_test_task1, predictions)
print(accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test_task1, predictions))