from sklearn import datasets
from pyspark import *
from pyspark.sql import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from sklearn.datasets import fetch_mldata
from pyspark.mllib.evaluation import MultilabelMetrics

import numpy as np
import random

#init spark
sc = SparkContext()
sqlContext = SQLContext(sc)
sc.setLogLevel('ERROR')

#
# DATA
#


mnist = fetch_mldata('MNIST original',data_home='/home/m2mocad/deleruyelle/Documents/MOCAD/TLDE/SparkTest/data')

data = mnist.data
labels = mnist.target

dataframe = [(int(labels[x]),Vectors.dense(data[x].tolist())) for x in range(len(data))]
random.shuffle(dataframe)

train = sqlContext.createDataFrame(dataframe[60000:], ["labels", "images"])
train.show(20)

test = sqlContext.createDataFrame(dataframe[:10000], ["labels", "images"])
test.show(20)


#
# CLASSIFIER
#

#logistic regression
lr = LogisticRegression(maxIter=500, regParam=0.05,featuresCol = "images",labelCol="labels",predictionCol = "predict_lr",rawPredictionCol = "rpc_lr",probabilityCol = "proba_lr")


# bayes classifier
cb = NaiveBayes(smoothing=1.0, modelType="multinomial",featuresCol = "images",labelCol="labels", predictionCol="predict_bc",rawPredictionCol = "rpc_bc",probabilityCol = "proba_bc")


#random forest
rf = RandomForestClassifier(numTrees=50,featuresCol="images",labelCol = "labels", predictionCol="predict_rf")

#multiplayer perceptron
layers = [784, 128, 128, 10]
mlp = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234, featuresCol="images",labelCol = "labels", predictionCol="predict_mlp")


#
# PIPELINE
#

pipeline = Pipeline(stages=[lr,cb,rf,mlp])
model = pipeline.fit(train)


#
# TEST MODEL
#


predict = model.transform(test)
predict.select("labels","predict_lr","predict_bc","predict_rf","predict_mlp").show(20)

#
# EVALUATION
#

evalCol= [("predict_lr","accuracy_lr"),("predict_bc","accuracy_bc"),("predict_rf","accuracy_rf"),("predict_mlp","accuracy_mlp")]
evaluation = [MulticlassClassificationEvaluator(labelCol="labels",predictionCol=evalCol[x][0],metricName="accuracy") for x in range(len(evalCol))]
[print(evalCol[x][1] +" : " + str(evaluation[x].evaluate(predict))) for x in range(len(evaluation))]
