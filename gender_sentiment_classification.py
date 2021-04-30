'''
    Gender classification and sentiment analysis from tweet data 
'''
# user binary crossentry for binary classification
from keras import layers
from keras import models
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import re
from textblob import TextBlob
import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# load data 
df = pd.read_csv("gender-classifier.csv")

# return the sentiment score of the tweet using TextBlob library
def get_sentiment_label(tweet):   
    # tweets sentiment analysis using Textblob
    analysis = TextBlob(tweet)
    score = analysis.sentiment.polarity
    if score > 0: 
        return 'positive'    
    elif score == 0: 
        return 'neutral' 
    else: 
        return 'negative'


# return the gender classification and sentiment label for each tweet 
def create_sentiment(file, df):   
    # open file and read the content in a list 
    with open(file, 'r') as filehandle:  
        week = file.replace(".txt","")
        tweets = [current_tweet.rstrip() for current_tweet in filehandle.readlines()]  
        for tweet in tweets: 
            # weeks.append(week)
            score = get_sentiment_label(tweet)
            # predict gender 
            tweet = [tweet]
            # predict the gender of the tweet based on the tweet content 
            label= nb.predict(vectorizer.transform(tweet))

            if label == 0: 
                gender = 'female'
            else: 
                gender = 'male'
            data = {'Week': [week], 'Content': [tweet], 'SentimentLabel':score, "Gender":gender}
            newData = pd.DataFrame(data)
            # append new row to the data
            df = df.append(newData, ignore_index = True)
    filename = './gender/'+str(week)+'.csv'
    # save to csv file 
    df.to_csv(filename)

    return df



# normalize the tect content
def normalize_text(text):
    # Remove non-ASCII chars.
    text = re.sub('[^\x00-\x7F]+',' ', text)
    # Remove URLs
    text = re.sub('https?:\/\/.*[\r\n]*', ' ', text) 
    # Remove special chars.
    text = re.sub('[?!+%{}:;.,"\'()\[\]_]', '',text)
    # Remove double spaces.
    text = re.sub('\s+',' ',text)
    
    return text

# gender classification training model
def train_gender_classification():
    # read the training set from Kaggle 
    df = pd.read_csv("./gender-classifier.csv")
    df[df["gender:confidence"] > 0.99]["gender"].value_counts()
    # data that has confidence of 0.99 or higher are included in the training set
    chosen_rows = df[df["gender"].isin(["male", "female"]) & (df["gender:confidence"] > 0.99)].index.tolist()
    n_samples = len(chosen_rows)
    random.shuffle(chosen_rows)
    test_size = round(n_samples*0.25)
    test_rows = chosen_rows[:test_size]
    val_rows = chosen_rows[test_size:2*test_size]
    train_rows = chosen_rows[2*test_size:]
    df["text_norm"] = [normalize_text(text) for text in df["text"]]

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(df.iloc[train_rows,:]["text_norm"])

    encoder = LabelEncoder()
    X_train = vectorizer.transform(df.loc[train_rows, "text_norm"])
    X_val = vectorizer.transform(df.loc[val_rows, "text_norm"])
    y_train = encoder.fit_transform(df.loc[train_rows, "gender"])
    y_val = encoder.transform(df.loc[val_rows, "gender"])
    # list(encoder.classes_)  ['female', 'male'] --> [0, 1]
    # use Naive Bayes MultinomialNB() classification to train the data
    nb = MultinomialNB()
    nb = nb.fit(X_train, y_train)
    # print(classification_report(y_val, nb.predict(X_val), target_names=encoder.classes_))
    accuracy_score(y_val, nb.predict(X_val))
  
    return nb, vectorizer

# predict the gender of the tweet based on tweet content using Naive Bayes Classification 
def predict_tweet_gender():
    # training the classification 
    nb, vectorizer= train_gender_classification()
    files = ['08Feb20.txt','15Feb20.txt','22Feb20.txt','01March20.txt','08March20.txt','15March20.txt','22March20.txt','01April20.txt','08April20.txt','15April20.txt','22April20.txt',
    '01May20.txt','08May20.txt','15May20.txt','22May20.txt','01June20.txt','08June20.txt','15June20.txt','22June20.txt','01July20.txt','08July20.txt','15July20.txt','22July20.txt',
    '01Aug20.txt','08Aug20.txt','15Aug20.txt','22Aug20.txt','01Sept20.txt','08Sept20.txt','15Sept20.txt','22Sept20.txt' ,'01Oct20.txt','08Oct20.txt','15Oct20.txt','22Oct20.txt',
    '01Nov20.txt','08Nov20.txt','15Nov20.txt','22Nov20.txt','01Dec20.txt','08Dec20.txt','15Dec20.txt','22Dec20.txt','01Jan21.txt','08Jan21.txt','15Jan21.txt','22Jan21.txt',
    '01Feb21.txt','08Feb21.txt','15Feb21.txt','22Feb21.txt']
  
    df= pd.DataFrame(columns = ('Week','Content', 'SentimentLabel'))

    for file in files: 
        d = create_sentiment(file,df)
        df = df.append(d, ignore_index = True)
    df.to_csv('./gender/total_gender_sentiment.csv')



if __name__ == "__main__":

    predict_tweet_gender()
    


    ################ group data in columns  ####################
    # data = pd.read_csv('./gender/total_gender_sentiment.csv')
    # data = data.groupby(['Gender','SentimentLabel'], sort=True).size().reset_index(name='Count')
    # print(data)
    female_neg_count = 54755
    female_pos_count = 147518
    female_neu_count = 154657
    male_neg_count = 83846
    male_pos_count = 264807
    male_neu_count = 249217
    female_neg_per = round(100 *female_neg_count/(female_neg_count+male_neg_count),2)
    female_neu_per = round(100 *female_neu_count/(female_neu_count+male_neu_count),2)
    female_pos_per = round(100 *female_pos_count/(female_pos_count+male_pos_count),2)

    male_neg_per = round(100 * male_neg_count/(female_neg_count+male_neg_count),2)
    male_neu_per = round(100 *male_neu_count/(female_neu_count+male_neu_count),2)
    male_pos_per = round(100 *male_pos_count/(female_pos_count+male_pos_count),2)
     # percentage of sentiment score based on Gender 
    female_percent = [female_neg_per, female_neu_per,female_pos_per]
    male_percent = [male_neg_per, male_neu_per,male_pos_per]
    
    
    print(female_percent)
    print(male_percent)
   
    
    male_percent = [male_neg_per, male_neu_per,male_pos_per]
    male_neg_per1 = round(100* male_neg_count/(male_neg_count+ male_pos_count+male_neu_count),2)
    male_neu_per1 = round(100* male_neu_count/(male_neg_count+ male_pos_count+male_neu_count),2)
    male_pos_per1 = round(100* male_pos_count/(male_neg_count+ male_pos_count+male_neu_count),2)

    female_neg_per1 = round(100* female_neg_count/(female_neg_count+ female_pos_count+female_neu_count),2)
    female_neu_per1 = round(100* female_neu_count/(female_neg_count+ female_pos_count+female_neu_count),2)
    female_pos_per1 = round(100* female_pos_count/(female_neg_count+ female_pos_count+female_neu_count),2)
   
    # percentage of gender by sentiment score 
    neg_percent = [female_neg_per1,male_neg_per1]
    neu_percent = [female_neu_per1,male_neu_per1]
    pos_percent = [female_pos_per1, male_pos_per1]
    print(neg_percent)
    print(neu_percent)
    print(pos_percent)


       
 

  
