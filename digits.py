from sklearn import datasets
from pyspark import *
from pyspark.sql import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier

import numpy as np

digits = datasets.load_digits()

data = digits.data[100:]
labels = digits.target[100:]

# 1797 x 2
datafr = []
for i in range(len(data)):
    datafr.append((int(labels[i]), Vectors.dense(data[i].tolist())))

sc = SparkContext()
sqlContext = SQLContext(sc)

# train data
train = sqlContext.createDataFrame(datafr, ["labels", "images"])
train.show(20)



lr = LogisticRegression(maxIter=3000, regParam=0.1,labelCol="labels",rawPredictionCol = "rPredict",featuresCol = "images",predictionCol = "p_nl",probabilityCol = "proba_nl")

"""
#multilayer perceptron
layers = [4, 15, 15, 15, 10]
lrNL = MultilayerPerceptronClassifier(maxIter=100, layers=layers)
"""


# bayes classifier
nb = NaiveBayes(smoothing=1.0, modelType="multinomial", predictionCol="p_nb", featuresCol = "images",labelCol="labels")


# create pipeline and fit
pipeline = Pipeline(stages=[lr,nb])
model = pipeline.fit(train)



data_test = digits.data[:100]
labels_test = digits.target[:100]

# 100 x 2
datatest = []
for i in range(len(data_test)):
    datatest.append((int(labels_test[i]), Vectors.dense(data_test[i].tolist())))

set_test = sqlContext.createDataFrame(datatest, ["labels", "images"])



# transform on test
prediction = model.transform(set_test)
prediction.show(50)
"""
probas = []
selected = prediction.select("label", "prediction")
selectedNL = predictionNL.select("label", "prediction")

probas.append(0)
for row in selected.collect():
    lbl, prediction = row
    if prediction == lbl:
        probas[0]+=1

probas.append(0)
for row in selectedNL.collect():
    lbl, prediction = row
    if prediction == lbl:
        probas[0]+=1
print(probas)
"""
