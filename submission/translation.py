from transformers import MarianMTModel, MarianTokenizer
from transliterate import Translit
from englishhinglish import isEnglish
from cleaning import clean_all


model_name = 'Helsinki-NLP/opus-mt-hi-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def hin2eng(hindi_text):
    hindi_tokens = tokenizer(hindi_text, return_tensors = "pt", padding = True)
    eng_tokens = model.generate(**hindi_tokens)
    return tokenizer.decode(eng_tokens[0], skip_special_tokens = True)

def isHindi(word):
    hindi_begin = b'\\u0900'
    hindi_end = b'\\u097F'
    if len(word)==0:
        return False
    unicode = word[0].encode('unicode-escape')
    if(unicode >= hindi_begin and unicode <= hindi_end) :
        return True
    else:
        return False



def transliterated(tweet_split):
    model_path = 'models/better2129Mar21.pt'
    transliterator = Translit(model_path)
    for i in range(len(tweet_split)):
        word = tweet_split[i]
        if len(word)>1:
            if (not isHindi(word)):
                if (not isEnglish(word)):
                    if(len(tweet_split[i]))>35:
                        tweet_split[i] = tweet_split[i][:35]
                    tweet_split[i] = transliterator.transliterate(tweet_split[i])
                    # print(tweet_split[i].encode('unicode-escape'))
    return tweet_split 

def engHin2Eng(tweet):
    tweet = clean_all(tweet)
    tweet_split = transliterated(tweet.split())
    
    # print(tweet_split)
    translated = ""
    hindi_flag = 0
    nonHindiCount = 0
    
    hindi_text = ""
    
    for word in tweet_split:
        if(isHindi(word)):
            if(hindi_flag):
                hindi_text+=word+" "
            else:
                hindi_flag=1
                hindi_text+=word+" "
        else:
            if(hindi_flag):
                nonHindiCount+=1
                hindi_text += word+" "
                if(nonHindiCount>=3):
                    translated += hin2eng(hindi_text) + " "
                    hindi_text = ""
                    hindi_flag=0
            else:
                translated += word + " "
    
    if(hindi_flag):
        translated += hin2eng(hindi_text)
        
    return translated