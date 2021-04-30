import glob
import os
import re
import json
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# Global Parameters
stop_words = set(stopwords.words('english'))

########### preprocess tweet data contents  ###############
content ="Great outcome #Covid-19 for our @UQ_News COVID-19 #vaccine securing additional $2M MRFF funding @ChappellDr @ProfPaulYoung @UQ_SCMB @AIBNatUQ Also great to see the continued Covid-19 investment @GregHuntMP!!\n$66 million for coronavirus-related research https://t.co/B7gOBtob7d"
#process data - tokenize data - remove links hashtag...
# works as expected 

def preprocess_tweet(tweet):
    # convert tweet contents to lowercase
    tweet.lower()
    # simplify the same words 
    tweet = re.sub('covid-19|covid19', 'coronavirus', tweet)
    tweet = re.sub('vaccines', 'vaccine', tweet)
        
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE) 
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
   
    # Remove common stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    
    # Apply stem words 
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    
    return " ".join(filtered_words)


# extract hashtag from the tweet
def hashtag_extract(tweet):
    hashtags=[]
    print(tweet)
    # tweet_tokens = word_tokenize()
   
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove punctuations
    # tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = tweet.split(' ')
    print(tweet)
    for i in tweet:
        print(i)
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    
    return hashtags

# read and save file to list for each week 
def read_save_content(file, name):
    tweets = []
    with open(file) as f : 
        for line in f: 
            lines = json.loads(line)
            tweet = lines['content']
            tweet = preprocess_tweet_text(tweet)
            tweets.append(tweet)

      # save content text to list file for each week 
    with open(name,'w') as file: 
        file.writelines("%s\n" % place for place in tweets)
    
     
def merge_json_files():
    result = []
    for f in glob.glob("*.json"):
        with open(f, "rb") as infile:
            result.append(json.load(infile))

    with open("merged_file.json", "wb") as outfile:
        json.dump(result, outfile)
    




# if __name__ == "__main__":
