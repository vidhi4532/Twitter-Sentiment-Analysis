import json 
import csv
import pandas as pd
import re
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#from preprocessor.api import clean
#nltk.download()
from nltk.tag import map_tag

fName="1.txt"
f=open("file.csv",'a',encoding="utf-8")
csvWriter=csv.writer(f)
tweets=[]
headers=['full_text','retweet_count','user','followers_count','place','coordinates','geo','created_at','id_str']
lem=WordNetLemmatizer()
pstem=PorterStemmer()
csvWriter.writerow(headers)
for line in open(fName,"r"):
    tweets.append(json.loads(line))

df=pd.DataFrame(data=tweets)
print(df.columns)
data=df.drop(['id', 'truncated',
     'display_text_range', 'entities', 'metadata', 'source',
       'in_reply_to_status_id', 'in_reply_to_status_id_str',
       'in_reply_to_user_id', 'in_reply_to_user_id_str',
       'in_reply_to_screen_name',
       'contributors', 'is_quote_status',
       'favorited', 'retweeted', 'lang',
       'quoted_status_id', 'quoted_status_id_str', 'extended_entities',
       'possibly_sensitive', 'quoted_status'],axis=1,inplace=True)
print(data)
print(df.columns)
df.to_csv("1.csv",header=True,index=False,encoding="utf-8")
count_lines=0
for tweet in tweets:
    try:
        csvWriter.writerow([tweet['full_text'],tweet['retweet_count'],tweet['user'],tweet['favourite_count'],tweet['place'],tweet['coordinates'],tweet['geo'],tweet['created_at'],str(tweet['id_str'])])
        count_lines+=1
    except Exception as e:
        print(e)
print(count_lines)
data=pd.read_csv("1.csv",encoding='unicode_escape')
print(data.head(5))
print(data.index)
print(len(data.index))
series=data.duplicated(["full_text"]).tolist()
print(series.count(True))
data=data.drop_duplicates(["full_text"])
data=data.drop_duplicates(["created_at","user"])
print(data)
data=data.drop(["place","coordinates","geo","id_str"],axis=1)
print(data.head(10))
print(data["full_text"][22])
data.drop(["retweet_count","favorite_count"],axis=1,inplace=True)
#data.drop(data.index[9869:10009],axis=0,inplace=True)
print(data.head(5))
print(data.columns)
#data.to_csv("2.csv",header=True,index=False,encoding="utf-8")

for i in range(len(data)):
    txt=data.iloc[i]["full_text"]
    txt=re.sub(r'^https?:\/\/.*[\r\n]*', ' ', str(txt))
    txt=re.sub(r'https?://[A-Za-z0-9./]+',' ',txt)
    txt=re.sub(r'[^a-zA-Z]',' ',txt)
    txt=re.sub(r"[^a-zA-Z]", ' ', txt)
    txt=re.sub(r'@[A-Z0-9a–z_:]+', ' ', txt)
    txt=re.sub(r'@[A-Za-z0–9]+', ' ', txt)
    txt=re.sub(r'[^0-9A-Za-z \t]',' ',txt)
    txt=re.sub(r'\w+:\/\/\S+',' ',txt)
    txt=re.sub(r'#', '', txt)
    txt=re.sub(r'@[^\s]+',' ',txt)
    txt=re.sub(r'@\w+',' ',txt)
    txt=re.sub(r'http://[^"\s]+',' ',txt)
    txt=re.sub("@BarackObama"," ",txt)
    #txt=p.clean(txt)
    txt = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', ' ', txt)
    data.at[i,"full_text"]=txt


#for i in range(len(data)):
    #txt=data.iloc[i]["full_text"]
    #print(data.index[i],txt)

def pos_tagging(df_copy):#takes
    li_swn=[]
    li_swn_pos=[]
    li_swn_neg=[]
    missing_words=[]
    for i in range(len(df_copy.index)):
        text = df_copy.loc[i]['full_text']
        tokens = nltk.word_tokenize(text)
        tagged_sent = nltk.pos_tag(tokens)
        store_it = [(word, map_tag('en-ptb', 'universal', tag)) for word, tag in tagged_sent]
        #print("Tagged Parts of Speech:",store_it)
        pos_total=0
        neg_total=0
        for word,tag in store_it:
            if(tag=='NOUN'):
                tag='n'
            elif(tag=='VERB'):
                tag='v'
            elif(tag=='ADJ'):
                tag='a'
            elif(tag=='ADV'):
                tag = 'r'
            else:
                tag='nothing'

            if(tag!='nothing'):
                concat = word+'.'+tag+'.01'
                try:
                    this_word_pos=swn.senti_synset(concat).pos_score()
                    this_word_neg=swn.senti_synset(concat).neg_score()
                    #print(word,tag,':',this_word_pos,this_word_neg)
                except Exception as e:
                    wor = lem.lemmatize(word)
                    concat = wor+'.'+tag+'.01'
                    # Checking if there's a possiblity of lemmatized word be accepted into SWN corpus
                    try:
                        this_word_pos=swn.senti_synset(concat).pos_score()
                        this_word_neg=swn.senti_synset(concat).neg_score()
                    except Exception as e:
                        wor = pstem.stem(word)
                        concat = wor+'.'+tag+'.01'
                        # Checking if there's a possiblity of lemmatized word be accepted
                        try:
                            this_word_pos=swn.senti_synset(concat).pos_score()
                            this_word_neg=swn.senti_synset(concat).neg_score()
                        except:
                            missing_words.append(word)
                            continue
                pos_total+=this_word_pos
                neg_total+=this_word_neg
        li_swn_pos.append(pos_total)
        li_swn_neg.append(neg_total)

        if(pos_total!=0 or neg_total!=0):
            if(pos_total>neg_total):
                li_swn.append(1)
            else:
                li_swn.append(-1)
        else:
            li_swn.append(0)
    df_copy.insert(3,"pos_score",li_swn_pos,True)
    df_copy.insert(4,"neg_score",li_swn_neg,True)
    df_copy.insert(5,"sent_score",li_swn,True)
    return df_copy



df=pos_tagging(data)
#print(df)

def lem_stem(df2):
    for i in range(len(df2.index)):
       text=df2.iloc[i]["full_text"]
       tokens=nltk.word_tokenize(text)
       tokens=[each_string.lower() for each_string in tokens]
       stop_words=stopwords.words('english')
       tokens=[word for word in tokens if word not in stop_words]
       for j in range(len(tokens)):
           tokens[j]=lem.lemmatize(tokens[j])
           tokens[j]=pstem.stem(tokens[j])
       tokens_sent=' '.join(tokens)
       df2.at[i,"full_text"]=tokens_sent
    return df2

df=lem_stem(df)
print(df)

df.to_csv("after_cleaning.csv",index=False)

