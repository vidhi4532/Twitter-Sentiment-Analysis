import tweepy
from textblob import TextBlob
import jsonpickle
import pandas as pd
import json

CONSUMER_KEY="GhSGCgT1bTwQYaAIk1bdjXAjA"
#print(CONSUMER_KEY)
CONSUMER_SECRET_KEY="dI4QsieZJ2dTSha3WXNhub1v1o9WpX3LpnC9x2jKWDmowWirts"
ACCESS_KEY="1286352826843578368-eEyjB67Z9lNA4MCgHcD8eNb1AMcblx"
ACCESS_SECRET_KEY="iVrthUmOfwmT0zLjArmAXwQSVjmlmxw2hmFmQauHYNhHd"

auth=tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET_KEY)
auth.set_access_token(ACCESS_KEY, ACCESS_SECRET_KEY)
auth.secure=True
api=tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
#print(api)
searchQuery="@BarackObama"
retweet_filter="-filter:retweets"
q=searchQuery+retweet_filter
tweetsPerQry=100
fName="1.txt"
sinceId=None

max_id=-1
maxTweets=10000

tweetCount=0
print("Downloading max {0} tweets".format(maxTweets))
with open(fName,"w") as f:
    while tweetCount < maxTweets:
        tweets=[]
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=q, lang ="en", count=tweetsPerQry, tweet_mode='extended')

                else:
                    new_tweets = api.search(q=q, lang ="en", count=tweetsPerQry,
                                        since_id=sinceId, tweet_mode='extended')
            else:
                if (not sinceId):
                    new_tweets = api.search(q=q, lang ="en", count=tweetsPerQry,
                                        max_id=str(max_id - 1), tweet_mode='extended')
                else:
                    new_tweets = api.search(q=q, lang ="en", count=tweetsPerQry,
                                        max_id=str(max_id - 1),
                                        since_id=sinceId, tweet_mode='extended')

            #print(new_tweets)

            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                #f.write(str(tweet.full_text.replace('\n','').encode("utf-8"))+"\n")
                f.write(jsonpickle.encode(tweet._json,unpicklable=False)+"\n")

            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
                
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break
                
print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))

