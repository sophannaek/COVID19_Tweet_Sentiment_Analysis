'''
Tweet Sentiment Analysis: 
1. The percentage of negative, positive, and neutral in the Political, Vaccine, Economic related tweets 
2. The percentage of negative, positive and neutral in the Political, Vaccine, Economic relatdy tweets by weekly , list the percentage of each category 
by each weekk 
3. relationship between ages/gender with the topic: 
    display among the ages/gender with the positive, negative or neural 


'''


import re
import json
from textblob import TextBlob
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import csv


def get_percent(file,df):
    weeks = []  
    vac_neg_count = vac_pos_count = vac_neu_count = eco_neg_count = eco_pos_count = eco_neu_count = 0
    pol_neg_count = pol_pos_count = pol_neu_count = vaccine_count = economic_count = politic_count = 0

    # open file and read the content in a list - process one week at a time 
    with open(file, 'r') as filehandle:  
        tweets = [current_tweet.rstrip() for current_tweet in filehandle.readlines()]  
        week = file.replace(".txt","")
        for i in range(0,3):
            weeks.append(week)

        for tweet in tweets: 
            # find its category, each tweet can fall into multiple category 
            group = get_category(tweet)
            print("group is ", group)
            if (group['vaccine'] != 0):
                vaccine_count += 1
                 # get its sentiment score -- increment the count for neg, pos, neu 
                score = get_sentiment_label(tweet)
                if score == 1: 
                    vac_pos_count += 1
                elif score == 0: 
                    vac_neu_count += 1
                else: 
                    vac_neg_count += 1 
            
            if (group['economy'] != 0):
                economic_count += 1
                 # get its sentiment score -- increment the count for neg, pos, neu 
                score = get_sentiment_label(tweet)
                if score == 1: 
                    eco_pos_count += 1
                elif score == 0: 
                    eco_neu_count += 1
                else: 
                    eco_neg_count += 1    

            if (group['politic'] != 0):
                politic_count += 1
                 # get its sentiment score -- increment the count for neg, pos, neu 
                score = get_sentiment_label(tweet)
                if score == 1: 
                    pol_pos_count += 1
                elif score == 0: 
                    pol_neu_count += 1
                else: 
                    pol_neg_count += 1 

    ###################   calculate the percent #####################
    neg_percent = pos_percent = neu_percent = []

    # calculate the vaccine group sentiment 
    if (vaccine_count == 0 ):
        vac_neg_percent = vac_pos_percent = vac_neu_percent = 0
    else:
        vac_neg_percent = round(100* vac_neg_count/vaccine_count,2)
        vac_pos_percent = round(100* vac_pos_count/vaccine_count,2)
        vac_neu_percent = round(100* vac_neu_count/vaccine_count,2)

    # calculate the econimic group sentiment 
    if economic_count == 0: 
        eco_neg_percent = eco_pos_percent = eco_neu_percent  = 0
    else:
        eco_neg_percent = round(100* eco_neg_count/economic_count,2)
        eco_pos_percent = round(100* eco_pos_count/economic_count,2)
        eco_neu_percent = round(100* eco_neu_count/economic_count,2)

    # calculate the politic group sentiment 
    if politic_count == 0:
        pol_neg_percent = pol_pos_percent = pol_neu_percent  = 0
    else: 
        pol_neg_percent = round(100* pol_neg_count/politic_count, 2)
        pol_pos_percent = round(100* pol_pos_count/politic_count, 2)
        pol_neu_percent = round(100* pol_neu_count/politic_count, 2)

    neg_percent.append(vac_neg_percent)
    neg_percent.append(eco_neg_percent)
    neg_percent.append(pol_neg_percent)

    pos_percent.append(vac_pos_percent)
    pos_percent.append(eco_pos_percent)
    pos_percent.append(pol_pos_percent)

    neu_percent.append(vac_neu_percent)
    neu_percent.append(eco_neu_percent)
    neu_percent.append(pol_neu_percent)
    
    category = ["vaccine", "economy", "political"]
    data = {"week": weeks, "category": category, "negative": neg_percent, "neutral": neu_percent, "positive": pos_percent}
    df = df.append(pd.DataFrame(data), ignore_index = True)


    return df
                


# count the given term in each period/weeks 
def get_category(tweet):
    term1 = ['logistic','distribution','vaccine']
    term2 =['job', 'jobless','economy','economic', 'stimulus','openning']
    term3 = ['election','political','compaign','elected','politics', 'political','politic']
    counts={"vaccine":0, "economy":0, "politic":0} 
 
    vaccine_flag = economy_flag = election_flag = False

    # tokenize the tweet
    tokens = tweet.split(" ")
    for token in tokens: 
        if token in term1 and vaccine_flag == False:
            counts['vaccine'] += 1
            vaccine_flag = True
  
        if token in term2 and economy_flag == False:
            counts['economy'] += 1
            economy_flag = True
        if token in term3 and election_flag == False:
            counts['politic'] += 1
            election_flag = True
    
    return counts



# return the sentiment score of the tweet using TextBlob library 
def get_sentiment_label(tweet):   
    # tweets sentiment analysis using Textblob
    analysis = TextBlob(tweet)
    score = analysis.sentiment.polarity
    if score > 0: 
        return 1
        
    elif score == 0: 
        return 0
        
    else: 
        return -1  




    
if __name__ == "__main__":
    # categorize each tweet by weekly basis 

    sentiment_data = pd.DataFrame(columns = ('week','category','negative','neutral','positive'))

    files = ['08Feb20.txt','15Feb20.txt','22Feb20.txt','01March20.txt','08March20.txt','15March20.txt','22March20.txt','01April20.txt','08April20.txt','15April20.txt','22April20.txt',
    '01May20.txt','08May20.txt','15May20.txt','22May20.txt','01June20.txt','08June20.txt','15June20.txt','22June20.txt','01July20.txt','08July20.txt','15July20.txt','22July20.txt',
    '01Aug20.txt','08Aug20.txt','15Aug20.txt','22Aug20.txt','01Sept20.txt','08Sept20.txt','15Sept20.txt','22Sept20.txt' ,'01Oct20.txt','08Oct20.txt','15Oct20.txt','22Oct20.txt',
    '01Nov20.txt','08Nov20.txt','15Nov20.txt','22Nov20.txt','01Dec20.txt','08Dec20.txt','15Dec20.txt','22Dec20.txt','01Jan21.txt','08Jan21.txt','15Jan21.txt','22Jan21.txt',
    '01Feb21.txt','08Feb21.txt','15Feb21.txt','22Feb21.txt']
    # files = ['08Feb20.txt','15Feb20.txt', '22Feb20.txt','01March20.txt']
    for file in files: 
        sentiment_data = get_percent(file,sentiment_data)
    print("--------------")
    print(sentiment_data)
    sentiment_data.to_csv('sentiment_data.csv',encoding='utf-8', index = False)

