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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
from sklearn.model_selection import cross_validate, KFold
import datetime
import time

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

X_train,X_val_test,y_train,y_val_test=train_test_split(X,y,test_size=0.1,random_state=SEED)
X_val,X_test,y_val,y_test=train_test_split(X_val_test,y_val_test,test_size=0.5,random_state=SEED)

#print(len(X_train),len(X_val_test),len(X_val))

classifiers=[MultinomialNB(),BernoulliNB(),AdaBoostClassifier(),
RidgeClassifier(),PassiveAggressiveClassifier(),Perceptron(),RandomForestClassifier()]


clf_names=['MultinomialNB()','BernoulliNB()','AdaBoostClassifier()',
'RidgeClassifier()','PassiveAggressiveClassifier()','Perceptron()','RandomForest Classifier()']

df=[]

for v in ['cv','tf']:
    for gram in range(1,4):
        i=0

        for clf in classifiers:
            
            if(clf=='RandomForest Classifier'):#special case
                clf = RandomForestClassifier(random_state=0,n_jobs=-1,class_weight="balanced")

            before = datetime.datetime.now()
            before = before.strftime("%H:%M:%S")
            start = time.time()
            
            if(v=='cv'):
                vec = TfidfVectorizer(ngram_range=(1,gram))
            else:
                vec = CountVectorizer(ngram_range=(1,gram))
                
            model = make_pipeline(vec,clf)
            model.fit(X_train.values.astype('U'),y_train.values.astype('U'))##
            labels = model.predict(X_val.values.astype('U'))
            ac = accuracy_score(y_val.values.astype('U'),labels)
            kfold = KFold(n_splits=10,shuffle=False,random_state=None)
            results = cross_validate(model,X_train.values.astype('U'),y_train.values.astype('U'),cv=kfold,return_train_score=True)
            crossval_test_score_mean=results['test_score'].mean()
            crossval_train_score_mean=results['train_score'].mean()
            crossval_test_score_std=results['test_score'].std()
            crossval_train_score_std=results['train_score'].std()
            after = datetime.datetime.now()
            after = after.strftime("%H:%M:%S")
            end = time.time()
            hours = int(after[0:2])-int(before[0:2])
            mins = int(after[3:5])-int(before[3:5])
            secs = int(after[6:8])-int(before[6:8])
            time_taken = str(hours)+":"+str(mins)+":"+str(secs)
            gr = str(gram)
            vec_gram = v+"_"+gr
            df.append([vec_gram,clf_names[i],ac,crossval_train_score_mean,crossval_test_score_mean,crossval_train_score_std,crossval_test_score_std, end-start])
            i+=1
            # data.append([0,0,0,0,0,0,0,0])
d = pd.DataFrame(df,columns=['Vec_Gram','Classifier','Ac','crossval_train_score_mean','crossval_test_score_mean','crossval_train_score_std','crossval_test_score_std','Time.2'])
print(d)