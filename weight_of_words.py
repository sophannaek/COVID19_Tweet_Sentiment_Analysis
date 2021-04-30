''' Get weight of keywords from the tweet contents from Feb 2020 - Feb 2021 using 
    the TfidVectorizer in sklearn library
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

 
def get_vectorize(contents):
    # Create TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),analyzer='word',max_features=30,min_df=20,max_df=0.7,use_idf=True)
     
    # Learn vocabulary in sentences. 
    vectorizer.fit_transform(contents).toarray()

    # Get feature name
    names = vectorizer.get_feature_names()
    vocab = vectorizer.vocabulary_
    # idf scores 
    idf = vectorizer.idf_
    data = dict(zip(names,idf))

    return data




####################### main ###################

if __name__ == "__main__":
    tweets = []

    # open file and read the content in a list
    with open('total_list.txt', 'r') as filehandle:
        tweets = [current_place.rstrip() for current_place in filehandle.readlines()]

    data = get_vectorize(tweets)
    token_weight = pd.DataFrame.from_dict(data, orient='index').reset_index()
    token_weight.columns=('topic','tdidf score')
    token_weight = token_weight.sort_values(by='tdidf score', ascending=False)

    # make a plot 
    sns.barplot(x='topic', y='tdidf score', data=token_weight)            
    plt.title("Weight per topic from Feb 2020 to March 2021")
    fig=plt.gcf()
    fig.set_size_inches(15,7)
    plt.xticks(rotation = 35)
    plt.savefig('weight_of_words.png')
    plt.show()
   


