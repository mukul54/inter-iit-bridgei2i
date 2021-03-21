import tweepy
import random
import pandas as pd
import time
import copy

# enter your search query here
LANGUAGE = 'en'
#Query list
query = [' BlackBerry Limited ', ' Honor ', ' Huawei ', ' OnePlus ', ' Oppo ', ' Realme ', ' Tecno ', ' Vivo ',  ' Xiaomi ', ' Zopo ', ' Verzo ',' Nokia ', ' Lenovo ', ' Karbonn Mobiles ', ' Lava ', ' HCL Technologies ', ' Jio ', ' LYF ', ' Micromax ', ' Spice ', ' Videocon ', ' Xolo ', ' Sony ', ' QMobile ',  ' LG ', ' Samsung ',' Acer ', ' Asus ', ' HTC ', 'Ericsson ', ' Apple ',  ' Google ', ' HP ', ' Motorola ']
for i in range(len(query)):
    query[i] +="Mobile"

# Set the file path 
if LANGUAGE =='en':
    CSV_PATH = 'sc_tweets.csv'
else:    
    CSV_PATH = 'sc_tweets_hindi.csv'
COL_NAME = 'Tweet'


# other setup variables
consumer_key = "W417cu6Oobhi5n2JwHzuDXfx1"
consumer_secret = "a9hkuQF1iETaBd5EJMiHluxtwXDSaThT2BFSELnPBi3DewCeSG"
access_key = "1596288438-3R1SBFnGb4F44pbfFt9sQFaucyRgiHUPILN8sbq"
access_secret = "rmJ3cjEiqaiXPvaLpddFAdtgb8CCeL6hxzVjTs30reHYZ"

#Scraping
for i in range(len(query)):
    #Authentication
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    # Change the 180 to any number; note that you will have to wait if the rate limit is exceeded
    cursor = tweepy.Cursor(api.search, q=query[i], tweet_mode='extended', lang=LANGUAGE)
    tweets = [x._json for x in cursor.items(180)]
    tweets = [tweet['full_text'] for tweet in tweets]
    tweets_df = pd.DataFrame(tweets, columns=[COL_NAME])

    #Saving the scraped data and removing duplicates if any
    try:
        d = pd.read_csv(CSV_PATH)
        d = pd.DataFrame(d[COL_NAME])

        print("Current len", len(d)," \nRemoving duplicate tweets")
        d = pd.concat([d,tweets_df])
        labels = d.duplicated()
        final = copy.copy(d[~labels])
        final.to_csv(CSV_PATH)
        print("Total till now: ", len(final))
        del final
        del d
    except:
        tweets_df.to_csv(CSV_PATH)
        print("Total till now: ", len(tweets_df))
    remaining = api.rate_limit_status()['resources']['search']['/search/tweets']['remaining']
    print("Remaining: ",remaining)
    
    #Setting delay time to avoid IP blocking XD
    if remaining < 30:
        print("Sleeping for 600s")
        time.sleep(600)
        continue
    #deleting variable to avoid large memory usage
    del auth
    del api
    del cursor
    del tweets
    del tweets_df

    t = random.randint(1,50)
    print("Waiting for ",t,"s")
    time.sleep(t)
