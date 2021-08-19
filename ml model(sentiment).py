import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
from sklearn.model_selection import cross_validate, KFold
import datetime
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


data=pd.read_csv("filesss.csv",index_col=False)
#print(data)

pos_text=""
neg_text=""
neut_text=""

for i in range(len(data.index)):
    if(data.iloc[i]["sent_score"]==1):
        pos_text+=str(data.iloc[i]["full_text"])
    elif(data.iloc[i]["sent_score"]==-1):
        neg_text+=str(data.iloc[i]["full_text"])
    else:
        neut_text+=str(data.iloc[i]["full_text"])
list_text = [pos_text,neg_text,neut_text]
#print(list_text)
#for text in list_text:
    #word_cloud=WordCloud(width=600,height=600,max_font_size=200).generate(text)
    #plt.figure(figsize=(12,10))
    #plt.imshow(word_cloud,interpolation="bilinear")
    #plt.axis("off")
    #plt.show()

SEED=4
X=data.full_text
y=data.sent_score

#x_train,x_val_test,y_train,y_val_test = train_test_split(X,y,test_size=0.1,random_state=SEED)
#x_val,x_test,y_val,y_test = train_test_split(x_val_test,y_val_test,test_size=0.5,random_state=SEED)
#print(len(X_train),len(X_val_test),len(X_val))
#print(x_traintf)

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer='word')),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier',LinearSVC()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

msg_train, msg_test, label_train, label_test=train_test_split(data['full_text'].values.astype('U'), data['sent_score'].values.astype('U'), test_size=0.2)
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions,label_test))
print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test))