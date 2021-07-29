import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from textblob import TextBlob
from pyspark.sql.functions import current_date
from spam_model.spam import spam_model
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, IndexToString, CountVectorizer, IndexToString,VectorIndexer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import PipelineModel
import string
import re
from pyspark.sql.functions import monotonically_increasing_id



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
    # Remove mentions  
    text = re.sub('@(\w+)',' ', text)
    # remove hashtags
    text = re.sub('#(\w+)',' ', text)
    return text

# get the sentiment analysis 
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


 
if __name__ == "__main__":

    MAX_MEMORY = "5g"
    # create Spark session
    spark = SparkSession.builder.appName("TwitterSentimentAnalysis")\
    .config("spark.executor.memory", MAX_MEMORY) \
    .config("spark.driver.memory", MAX_MEMORY) \
    .getOrCreate()

    # # text preprocesisng 
    text_preprocessing_udf = udf(normalize_text, StringType())
    # get sentiment analysis
    tweet_sentiment_udf = udf(get_sentiment_label, StringType())

    # load the pre-trained model 
    model = PipelineModel.load('./sample-model')
    
    try: 
        # # read the tweet data from socket -- lines is dataframe
        lines = spark.readStream.format("socket").option("host", "localhost").option("port", 5555).load()
        lines.printSchema()
        # preprocessing
        tweets_df = lines.withColumn("Cleaned Tweet", text_preprocessing_udf(lines['value']))
        # sentiment analysis
        tweets_df = tweets_df.withColumn('Sentiment', tweet_sentiment_udf(tweets_df['Cleaned Tweet']))

        # predictions
        predictions = model.transform(tweets_df)
       
        # # print to console -- 
        query = prediction.writeStream.format("console").outputMode("append").option("truncate", "true").start().awaitTermination()

        # write to json files 
        # data_sink = predictions.writeStream.format("json").option("path", "./checkpoint").option("checkpointLocation","./checkpoint").start()
        # data_sink.awaitTermination()
    
    except: 
        print("the stream can not be read!")
    