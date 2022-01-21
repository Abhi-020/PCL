# PCL

This is the implementation repostiory for the completition [PCL](https://competitions.codalab.org/competitions/34344).

Timeline:

```
Evaluation Start: January 10, 2022
Evaluation End: January 31, 2022
Paper submissions due: February 23, 2022
Notification to authors: March 31, 2022

```
# Experiment in Progress


|Models|Description|Precion|Recall|weighted average F1|Accuracy F1|Remark|
|---|---|---|---|---|---|---|
|GaussianNB|TFIDF Feature|.70|.69|.69|68.7|partial towards class 0|
|GaussianNB|CV Feature|.75|.21|.30|21.4||
|MultinomialNB|TFIDF Feature|.67|.82|.73|81.6|Highly biased|
|MultinomialNB|CV Feature|.74|.76|.75|76.2|1. overall accc,pre,re is stable throughout.2.class 2 doesn,t given any weightage|
|Logistic|CV Feature|.87|.89|.88|89|CM matrix 0,0 value very in comparison to other three value|
|Logistic|TFIDF Feature|.87|.90|.86|90|Confusion matrix unevenly distributed|
|Logistic|CV Feature,weighted|.87|.80|.83|.80|cm well distributed|
|Logistic|TFIDF, Weighted|.89|.87|.88|87| Cm Well distributed
|Logistic|TFIDF, Weighted(W2V)|.89|.88|.88|.88| Cm Well distributed|
|Logistic|CV, Weighted(W2V)|.88|.79|.83|.79| Cm Well distributed|
|SVM|CV,Weighted|.88|.84|.85|84|contribution from both classes|
|SVM|CV,W2V|.87|.83|.85|83| CM well distributed|
|MLP|CV,W2V|.87|.89|.87|89|
|MLP|TFIDF,W2V|.88|.90|.88|90|
|SGD|CV,W2V|.86|.88|.87|88|
|SGD Classifier|||||||
|XGBoost |||||||

# current ToDo's
- [ ] Bert MLP weighted.(1)
- [ ] Bert Bi-lstm [attention(wt)](5) (with Or without weighted)
- [ ] Fix dataset Imbalance (Aman)
- [ ] Bert, XLM, GPT, Distill- bert, roberta(hugging Face)(2)
- [ ] Toxic Bert, News Bert (hugging Face)(4)
- [ ] Task-2 data analysis (3)
- [ ] Task-2 Model - logistic, MLP, Bi-lstm,  Bert (2) note input to the model : sentence + span interval (maybe + span again)**
- [ ] Adversial training (Aman)
- [ ] B-LSTM (Aman)
# To do

- [ ] Implement Baseline

|MODELS|Ready|
|---|---|
|Logistic Regression |x|
|Naive Bayes|x|
|SVM |x|
|MLP|x|
|SGD Classifier|x|
|XGBoost |x|
|Bi-LSTM ||

- [ ] Features: countvectorizer, tfidfvectorizer, word2vec (local), gloVe, elmo, BERT

- [ ] Discuss data analysis

- [ ] List-down main models  

|MODELS|F1|
|---|---|
|BERT ||
