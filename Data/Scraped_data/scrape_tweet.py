import tweepy
import csv
import random
import pandas as pd
import time
# enter your search query here
query = [' BlackBerry Limited ', ' Honor ', ' Huawei ', ' OnePlus ', ' Oppo ', ' Realme ', ' Tecno ', ' Vivo ',  ' Xiaomi ', ' Zopo ', ' Verzo ',' Nokia ', ' Lenovo ', ' Karbonn Mobiles ', ' Lava ', ' HCL Technologies ', ' Jio ', ' LYF ', ' Micromax ', ' Spice ', ' Videocon ', ' Xolo ', ' Sony ', ' QMobile ',  ' LG ', ' Samsung ',' Acer ', ' Asus ', ' HTC ', 'Ericsson ', ' Apple ',  ' Google ', ' HP ', ' Motorola ']
for i in range(len(query)):
    query[i] +="Mobile"
consumer_key = "W417cu6Oobhi5n2JwHzuDXfx1"
consumer_secret = "a9hkuQF1iETaBd5EJMiHluxtwXDSaThT2BFSELnPBi3DewCeSG"
access_key = "1596288438-3R1SBFnGb4F44pbfFt9sQFaucyRgiHUPILN8sbq"
access_secret = "rmJ3cjEiqaiXPvaLpddFAdtgb8CCeL6hxzVjTs30reHYZ"
while 1:
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    # Change the 180 to any number; note that you will have to wait if the rate limit is exceeded
    cursor = tweepy.Cursor(api.search, q=random.choice(query), tweet_mode='extended', lang='en')
    tweets = [x._json for x in cursor.items(180)]
    tweets = [tweet['full_text'] for tweet in tweets]
    tweets_df = pd.DataFrame(tweets)
    tweets_df.to_csv("sc_tweets.csv",  mode='a')
    d = pd.read_csv('sc_tweets.csv')
    print("Remaining: ",api.rate_limit_status()['resources']['search']['/search/tweets']['remaining'])
    print("Total till now: ", len(d))
    del auth
    del api
    del cursor
    del tweets
    del tweets_df
    del d
    t = random.randint(20,300)
    print("Waiting for ",t,"s")
    time.sleep(t)
