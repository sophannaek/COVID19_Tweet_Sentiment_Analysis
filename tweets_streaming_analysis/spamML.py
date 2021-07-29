'''
Machine learning model for spam detection on tweet content.
TF-IDF vectorizer for training the model 
'''

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, IndexToString, CountVectorizer, IndexToString,VectorIndexer
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import PipelineModel
import string
import re
from pyspark.sql.functions import monotonically_increasing_id


# normalize the text content
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

# load, train, evaluate ,and save model
if __name__ == "__main__":
    MAX_MEMORY = "5g"
    # create Spark session
    spark = SparkSession.builder.appName("TwitterSentimentAnalysis")\
    .config("spark.executor.memory", MAX_MEMORY) \
    .config("spark.driver.memory", MAX_MEMORY) \
    .getOrCreate()

    # load data from csv file
    df = spark.read.format('com.databricks.spark.csv') \
                .options(header='true', inferschema='true') \
                .load("./dataset/train.csv",header=True)       
    df.printSchema()
    df.show(5)

    # # text preprocesisng 
    text_preprocessing_udf = udf(normalize_text, StringType())
    # normalizing text 
    df = df.withColumn("Cleaned Tweet",text_preprocessing_udf(df["Tweet"]))
    # create user defined function 

    # feature extraction 
    df = df.drop('_c7','actions','is_retweet','following','followers','location')
    # remove any missing values 
    df = df.dropna()
    # Create Unique ID
    # df = df.withColumn("uid", monotonically_increasing_id())


    # convert label to numeric value
    stringIndexer = StringIndexer(inputCol="Type", outputCol="label")
    # label_encoder = stringIndexer.fit(df).setHandleInvalid("skip")
    label_encoder = stringIndexer.fit(df)
    df = label_encoder.transform(df)
    df = df.drop('Type')
    df.show(5)

    # Split the data into training and test sets (25% held out for testing)
    (trainingData, testData) = df.randomSplit([0.75, 0.25])

    # # train the data using TF-IDF vectorizer
    # Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and nb.
    tokenizer = Tokenizer(inputCol="Cleaned Tweet", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures")
    vectorizer = CountVectorizer(inputCol= "words", outputCol="rawFeatures")
    # idf = IDF(minDocFreq=3, inputCol="rawFeatures", outputCol="features")
    idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features", minDocFreq=5)
    # Naive Bayes model
    nb = NaiveBayes(featuresCol='features', labelCol='label')
    # convert the numeric label back to string label
    categoryConverter = IndexToString().setInputCol("prediction").setOutputCol("category").setLabels(label_encoder.labels)
    # Pipeline Architecture
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nb, label_encoder, categoryConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    ## prediction
    predictions = model.transform(testData)
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Accuracy = %g" % accuracy)
    print("Test Error = %g" % (1.0 - accuracy))
    predictions.select("Cleaned Tweet","label","prediction").show()

     # save PipelineModel
    model.write().overwrite().save('sample-model')
    
   