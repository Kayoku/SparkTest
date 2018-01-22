from sklearn import datasets
from pyspark import *
from pyspark.sql import *
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

set_training = sqlContext.createDataFrame(datafr, ["label", "features"])
lr = LogisticRegression(maxIter=3000, regParam=0.1)
layers = [4, 15, 15, 15, 10]
lrNL = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)

lrModelNL = lrNL.fit(set_training)
lrModel = lr.fit(set_training)

data_test = digits.data[:100]
labels_test = digits.target[:100]

# 100 x 2
datatest = []
for i in range(len(data_test)):
    datatest.append((int(labels_test[i]), Vectors.dense(data_test[i].tolist())))

set_test = sqlContext.createDataFrame(datatest, ["label", "features"])
prediction = lrModel.transform(set_test)
predictionNL = lrModelNL.transform(set_test)

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
