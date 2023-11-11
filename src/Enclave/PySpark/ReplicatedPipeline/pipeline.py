from pyspark.sql.functions import col,isnan, when, datediff, count, flatten, concat
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator


import pandas as pd
from pyspark.ml import Pipeline

from pyspark.sql.functions import sum, first,  countDistinct, max
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.mllib.stat import Statistics 
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql import Window
import pyspark.sql.functions as f


from sklearn.metrics import confusion_matrix

@transform_pandas(
    Output(rid="ri.vector.main.execute.9c112bf0-1e94-4cf6-9a14-83d7311ba419"),
    eraPrep=Input(rid="ri.foundry.main.dataset.341c241d-8523-4742-afa6-213415ddc774")
)
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
def GradientBoosting(eraPrep):
    df = eraPrep

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)  #death is 1
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    # Alive = 1, Deceased = 0
    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)  #alive is 1
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)
    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)
    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)
    
    #Deceased label model
    gbtDeceased = GBTClassifier(labelCol="deceased", featuresCol="features", maxIter=10)
    modelD = gbtDeceased.fit(trainD)

    predictionsDeceased = modelD.transform(testD)

    gbtAlive = GBTClassifier(labelCol="alive", featuresCol="features", maxIter=10)
    modelA = gbtAlive.fit(trainA)

    predictionsAlive = modelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_evalA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
    
    resultsD = modelD.transform(testD)
    print("\n[Performance on Deceased Label Test set]")
    printPerformanceD(resultsD)

    resultsA = modelA.transform(testA)
    print("\n[Performance on Alive Label Test set]")
    printPerformanceA(resultsA)

    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.87578a0a-9e6f-452a-8c00-68ee0c96b106"),
    condPrep=Input(rid="ri.foundry.main.dataset.d03ab161-fabe-4425-b2e6-63474c6e6a11")
)
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
def GradientBoosting2(condPrep):
    df = condPrep

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)  #death is 1
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    # Alive = 1, Deceased = 0
    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)  #alive is 1
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)
    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)
    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)
    
    #Deceased label model
    gbtDeceased = GBTClassifier(labelCol="deceased", featuresCol="features", maxIter=10)
    modelD = gbtDeceased.fit(trainD)

    predictionsDeceased = modelD.transform(testD)

    gbtAlive = GBTClassifier(labelCol="alive", featuresCol="features", maxIter=10)
    modelA = gbtAlive.fit(trainA)

    predictionsAlive = modelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_evalA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
    
    resultsD = modelD.transform(testD)
    print("\n[Performance on Deceased Label Test set]")
    printPerformanceD(resultsD)

    resultsA = modelA.transform(testA)
    print("\n[Performance on Alive Label Test set]")
    printPerformanceA(resultsA)

    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.28789c0d-29ee-4b65-9224-18f7ff409f5c"),
    obs=Input(rid="ri.foundry.main.dataset.20a84953-6c28-4d22-94ac-20105a7ff051")
)
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
def GradientBoosting3(obs):
    df = obs

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)  #death is 1
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    # Alive = 1, Deceased = 0
    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)  #alive is 1
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)
    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)
    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)
    
    #Deceased label model
    gbtDeceased = GBTClassifier(labelCol="deceased", featuresCol="features", maxIter=10)
    modelD = gbtDeceased.fit(trainD)

    predictionsDeceased = modelD.transform(testD)

    gbtAlive = GBTClassifier(labelCol="alive", featuresCol="features", maxIter=10)
    modelA = gbtAlive.fit(trainA)

    predictionsAlive = modelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_evalA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
    
    resultsD = modelD.transform(testD)
    print("\n[Performance on Deceased Label Test set]")
    printPerformanceD(resultsD)

    resultsA = modelA.transform(testA)
    print("\n[Performance on Alive Label Test set]")
    printPerformanceA(resultsA)

    return resultsD

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e776b66d-94f8-43df-bbef-127aff5f75af"),
    expoPrep=Input(rid="ri.foundry.main.dataset.acf6f7df-4896-405c-82e4-3b556f8fe6cf")
)
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
def GradientBoosting4(expoPrep):
    df = expoPrep

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)  #death is 1
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    # Alive = 1, Deceased = 0
    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)  #alive is 1
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)
    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)
    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)
    
    #Deceased label model
    gbtDeceased = GBTClassifier(labelCol="deceased", featuresCol="features", maxIter=10)
    modelD = gbtDeceased.fit(trainD)

    predictionsDeceased = modelD.transform(testD)

    gbtAlive = GBTClassifier(labelCol="alive", featuresCol="features", maxIter=10)
    modelA = gbtAlive.fit(trainA)

    predictionsAlive = modelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_evalA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
    
    resultsD = modelD.transform(testD)
    print("\n[Performance on Deceased Label Test set]")
    printPerformanceD(resultsD)

    resultsA = modelA.transform(testA)
    print("\n[Performance on Alive Label Test set]")
    printPerformanceA(resultsA)

    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.3d1defec-e616-438a-ad9c-7abe7f877c82"),
    drugPrep=Input(rid="ri.foundry.main.dataset.1914a0ea-151f-49bc-9981-e2ffb0ac9aae")
)
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
def GradientBoosting5(drugPrep):
    df = drugPrep

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)  #death is 1
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    # Alive = 1, Deceased = 0
    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)  #alive is 1
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)
    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)
    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)
    
    #Deceased label model
    gbtDeceased = GBTClassifier(labelCol="deceased", featuresCol="features", maxIter=10)
    modelD = gbtDeceased.fit(trainD)

    predictionsDeceased = modelD.transform(testD)

    gbtAlive = GBTClassifier(labelCol="alive", featuresCol="features", maxIter=10)
    modelA = gbtAlive.fit(trainA)

    predictionsAlive = modelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_evalA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
    
    resultsD = modelD.transform(testD)
    print("\n[Performance on Deceased Label Test set]")
    printPerformanceD(resultsD)

    resultsA = modelA.transform(testA)
    print("\n[Performance on Alive Label Test set]")
    printPerformanceA(resultsA)

    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.9631f85f-e013-465e-ae45-c9560fe36744"),
    condition_era_samp=Input(rid="ri.foundry.main.dataset.14761745-9012-429c-8933-73e0aae4dfb4")
)

def INFO(condition_era_samp):
    df = condition_era_samp
    conDF = df.groupby('condition_concept_id').count()
    df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).toPandas().head()
    return conDF

@transform_pandas(
    Output(rid="ri.vector.main.execute.01a8faed-2493-49b3-bbc1-d36af09e2306"),
    eraPrep=Input(rid="ri.foundry.main.dataset.341c241d-8523-4742-afa6-213415ddc774")
)
from pyspark.ml.classification import LogisticRegression
import numpy as np
from pyspark.mllib.evaluation import RegressionMetrics

def LogRegressionModel(eraPrep):
    df = eraPrep

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)

    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)

    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)

    lrD = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'deceased', featuresCol = 'features')
    lrModelD = lrD.fit(trainD)
    resultsD = lrModelD.transform(testD)
    
    lrA = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'alive', featuresCol = 'features')
    lrModelA = lrA.fit(trainA)
    resultsA = lrModelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    print('Deceased label')
    print('Accuracy: {:0.2f}'.format(lrModelD.evaluate(testD).accuracy))
    print("Coefficients: ", lrModelD.coefficients)
    print("Intercept: ", lrModelD.intercept)
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    resultsD = lrModelD.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceD(resultsD)

    print('Alive label')
    print('Accuracy: {:0.2f}'.format(lrModelA.evaluate(testA).accuracy))
    print("Coefficients: ", lrModelA.coefficients)
    print("Intercept: ", lrModelA.intercept)
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
        y_test = np.array(resultsA.select("deceased").collect()).flatten()
        y_pred = np.array(resultsA.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
    resultsA = lrModelA.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceA(resultsA)
    
    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.4c7952ec-c280-42f2-b703-41e971fb444f"),
    condPrep=Input(rid="ri.foundry.main.dataset.d03ab161-fabe-4425-b2e6-63474c6e6a11")
)
from pyspark.ml.classification import LogisticRegression
import numpy as np
from pyspark.mllib.evaluation import RegressionMetrics

def LogRegressionModel2(condPrep):
    df = condPrep

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)

    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)

    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)

    lrD = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'deceased', featuresCol = 'features')
    lrModelD = lrD.fit(trainD)
    resultsD = lrModelD.transform(testD)
    
    lrA = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'alive', featuresCol = 'features')
    lrModelA = lrA.fit(trainA)
    resultsA = lrModelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    print('Deceased label')
    print('Accuracy: {:0.2f}'.format(lrModelD.evaluate(testD).accuracy))
    print("Coefficients: ", lrModelD.coefficients)
    print("Intercept: ", lrModelD.intercept)
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    resultsD = lrModelD.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceD(resultsD)

    print('Alive label')
    print('Accuracy: {:0.2f}'.format(lrModelA.evaluate(testA).accuracy))
    print("Coefficients: ", lrModelA.coefficients)
    print("Intercept: ", lrModelA.intercept)
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
        y_test = np.array(resultsA.select("deceased").collect()).flatten()
        y_pred = np.array(resultsA.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
    resultsA = lrModelA.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceA(resultsA)
    
    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.421bde84-341a-4397-97b9-afeec99119e5"),
    obs=Input(rid="ri.foundry.main.dataset.20a84953-6c28-4d22-94ac-20105a7ff051")
)
from pyspark.ml.classification import LogisticRegression
import numpy as np
from pyspark.mllib.evaluation import RegressionMetrics

def LogRegressionModel3(obs):
    df = obs

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)

    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)

    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)

    lrD = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'deceased', featuresCol = 'features')
    lrModelD = lrD.fit(trainD)
    resultsD = lrModelD.transform(testD)
    
    lrA = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'alive', featuresCol = 'features')
    lrModelA = lrA.fit(trainA)
    resultsA = lrModelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    print('Deceased label')
    print('Accuracy: {:0.2f}'.format(lrModelD.evaluate(testD).accuracy))
    print("Coefficients: ", lrModelD.coefficients)
    print("Intercept: ", lrModelD.intercept)
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    resultsD = lrModelD.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceD(resultsD)

    print('Alive label')
    print('Accuracy: {:0.2f}'.format(lrModelA.evaluate(testA).accuracy))
    print("Coefficients: ", lrModelA.coefficients)
    print("Intercept: ", lrModelA.intercept)
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
        y_test = np.array(resultsA.select("deceased").collect()).flatten()
        y_pred = np.array(resultsA.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
    resultsA = lrModelA.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceA(resultsA)
    
    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.d79db5be-ffca-4622-9caf-b012aa3a4697"),
    expoPrep=Input(rid="ri.foundry.main.dataset.acf6f7df-4896-405c-82e4-3b556f8fe6cf")
)
from pyspark.ml.classification import LogisticRegression
import numpy as np
from pyspark.mllib.evaluation import RegressionMetrics

def LogRegressionModel4(expoPrep):
    df = ecpoPrep

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)

    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)

    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)

    lrD = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'deceased', featuresCol = 'features')
    lrModelD = lrD.fit(trainD)
    resultsD = lrModelD.transform(testD)
    
    lrA = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'alive', featuresCol = 'features')
    lrModelA = lrA.fit(trainA)
    resultsA = lrModelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    print('Deceased label')
    print('Accuracy: {:0.2f}'.format(lrModelD.evaluate(testD).accuracy))
    print("Coefficients: ", lrModelD.coefficients)
    print("Intercept: ", lrModelD.intercept)
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    resultsD = lrModelD.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceD(resultsD)

    print('Alive label')
    print('Accuracy: {:0.2f}'.format(lrModelA.evaluate(testA).accuracy))
    print("Coefficients: ", lrModelA.coefficients)
    print("Intercept: ", lrModelA.intercept)
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
        y_test = np.array(resultsA.select("deceased").collect()).flatten()
        y_pred = np.array(resultsA.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
    resultsA = lrModelA.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceA(resultsA)
    
    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.b3d825f6-1170-454f-a4d5-21ea9add8ee2"),
    drugPrep=Input(rid="ri.foundry.main.dataset.1914a0ea-151f-49bc-9981-e2ffb0ac9aae")
)
from pyspark.ml.classification import LogisticRegression
import numpy as np
from pyspark.mllib.evaluation import RegressionMetrics

def LogRegressionModel5(drugPrep):
    df = drugPrep

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)

    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)

    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)

    lrD = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'deceased', featuresCol = 'features')
    lrModelD = lrD.fit(trainD)
    resultsD = lrModelD.transform(testD)
    
    lrA = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'alive', featuresCol = 'features')
    lrModelA = lrA.fit(trainA)
    resultsA = lrModelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    print('Deceased label')
    print('Accuracy: {:0.2f}'.format(lrModelD.evaluate(testD).accuracy))
    print("Coefficients: ", lrModelD.coefficients)
    print("Intercept: ", lrModelD.intercept)
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    resultsD = lrModelD.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceD(resultsD)

    print('Alive label')
    print('Accuracy: {:0.2f}'.format(lrModelA.evaluate(testA).accuracy))
    print("Coefficients: ", lrModelA.coefficients)
    print("Intercept: ", lrModelA.intercept)
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
        y_test = np.array(resultsA.select("deceased").collect()).flatten()
        y_pred = np.array(resultsA.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
    resultsA = lrModelA.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceA(resultsA)
    
    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.8260efa9-4863-4500-a1f0-6fd7ceaa1770"),
    UnivariateFeatureSelector2=Input(rid="ri.vector.main.execute.8f64eba0-5c8a-41c6-97cd-6099e723678f")
)
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import RegressionMetrics

def LogSelectedFeaturesModel(UnivariateFeatureSelector2):
    df = UnivariateFeatureSelector2

    # stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1)

    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, labelCol = 'deceased', featuresCol = 'selectedFeatures')
    lrModel = lr.fit(train)
    result = lrModel.transform(test)
    
    print('Accuracy: {:0.2f}'.format(lrModel.evaluate(test).accuracy))
    print("Coefficients: ", lrModel.coefficients)
    print("Intercept: ", lrModel.intercept)
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.fb147727-6006-4c4b-bdf2-cadf58c30e05"),
    condPrep=Input(rid="ri.foundry.main.dataset.d03ab161-fabe-4425-b2e6-63474c6e6a11")
)
from pyspark.ml.classification import RandomForestClassifier
import numpy as np

def RandomForest2(condPrep):
    #parameters
    maxCategories=4
    seed=42

    df = condPrep
    label = 'deceased'    

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)

    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)

    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)

    rfD = RandomForestClassifier(labelCol = 'deceased', featuresCol = 'features')
    rfA = RandomForestClassifier(labelCol = 'alive', featuresCol = 'features')    
    
    
    lrModelD = rfD.fit(trainD)
    resultsD = lrModelD.transform(testD)

    lrModelA = rfA.fit(trainA)
    resultsA = lrModelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    print('Deceased label')
    print('Accuracy: {:0.2f}'.format(lrModelD.evaluate(testD).accuracy))
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    resultsD = lrModelD.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceD(resultsD)

    print('Alive label')
    print('Accuracy: {:0.2f}'.format(lrModelA.evaluate(testA).accuracy))
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
        y_test = np.array(resultsA.select("deceased").collect()).flatten()
        y_pred = np.array(resultsA.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
    resultsA = lrModelA.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceA(resultsA)
    
    return resultsD

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7bf4046c-9e23-48eb-9a78-a3dbd43e866f"),
    obs=Input(rid="ri.foundry.main.dataset.20a84953-6c28-4d22-94ac-20105a7ff051")
)
from pyspark.ml.classification import RandomForestClassifier
import numpy as np

def RandomForest3(obs):
    #parameters
    maxCategories=4
    seed=42

    df = obs
    label = 'deceased'    

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)

    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)

    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)

    rfD = RandomForestClassifier(labelCol = 'deceased', featuresCol = 'features')
    rfA = RandomForestClassifier(labelCol = 'alive', featuresCol = 'features')    
    
    
    lrModelD = rfD.fit(trainD)
    resultsD = lrModelD.transform(testD)

    lrModelA = rfA.fit(trainA)
    resultsA = lrModelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    print('Deceased label')
    print('Accuracy: {:0.2f}'.format(lrModelD.evaluate(testD).accuracy))
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    resultsD = lrModelD.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceD(resultsD)

    print('Alive label')
    print('Accuracy: {:0.2f}'.format(lrModelA.evaluate(testA).accuracy))
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
        y_test = np.array(resultsA.select("deceased").collect()).flatten()
        y_pred = np.array(resultsA.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
    resultsA = lrModelA.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceA(resultsA)
    
    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.ca811206-918a-411c-920c-25e72ca445fa"),
    expoPrep=Input(rid="ri.foundry.main.dataset.acf6f7df-4896-405c-82e4-3b556f8fe6cf")
)
from pyspark.ml.classification import RandomForestClassifier
import numpy as np

def RandomForest4(expoPrep):
    #parameters
    maxCategories=4
    seed=42

    df = expoPrep
    label = 'deceased'    

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)

    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)

    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)

    rfD = RandomForestClassifier(labelCol = 'deceased', featuresCol = 'features')
    rfA = RandomForestClassifier(labelCol = 'alive', featuresCol = 'features')    
    
    
    lrModelD = rfD.fit(trainD)
    resultsD = lrModelD.transform(testD)

    lrModelA = rfA.fit(trainA)
    resultsA = lrModelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    print('Deceased label')
    print('Accuracy: {:0.2f}'.format(lrModelD.evaluate(testD).accuracy))
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    resultsD = lrModelD.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceD(resultsD)

    print('Alive label')
    print('Accuracy: {:0.2f}'.format(lrModelA.evaluate(testA).accuracy))
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
        y_test = np.array(resultsA.select("deceased").collect()).flatten()
        y_pred = np.array(resultsA.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
    resultsA = lrModelA.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceA(resultsA)
    
    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.89d159ad-552a-4ff8-aadf-fdff3ec63e62"),
    drugPrep=Input(rid="ri.foundry.main.dataset.1914a0ea-151f-49bc-9981-e2ffb0ac9aae")
)
from pyspark.ml.classification import RandomForestClassifier
import numpy as np

def RandomForest5(drugPrep):
    #parameters
    maxCategories=4
    seed=42

    df = drugPrep
    label = 'deceased'    

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)

    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)

    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)

    rfD = RandomForestClassifier(labelCol = 'deceased', featuresCol = 'features')
    rfA = RandomForestClassifier(labelCol = 'alive', featuresCol = 'features')    
    
    
    lrModelD = rfD.fit(trainD)
    resultsD = lrModelD.transform(testD)

    lrModelA = rfA.fit(trainA)
    resultsA = lrModelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    print('Deceased label')
    print('Accuracy: {:0.2f}'.format(lrModelD.evaluate(testD).accuracy))
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    resultsD = lrModelD.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceD(resultsD)

    print('Alive label')
    print('Accuracy: {:0.2f}'.format(lrModelA.evaluate(testA).accuracy))
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
        y_test = np.array(resultsA.select("deceased").collect()).flatten()
        y_pred = np.array(resultsA.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
    resultsA = lrModelA.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceA(resultsA)
    
    return resultsD

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.148be3b6-b676-4b1c-b6c1-3a255e2e3f41"),
    eraPrep=Input(rid="ri.foundry.main.dataset.341c241d-8523-4742-afa6-213415ddc774")
)
from pyspark.ml.classification import RandomForestClassifier
import numpy as np

def RandomForestCond(eraPrep):
    #parameters
    maxCategories=4
    seed=42

    df = eraPrep
    label = 'deceased'    

    # stratified split
    classD0 = df.filter(df["deceased"]==0)
    classD1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", classD0.count())
    print("Class 1 (deceased= 1): ", classD1.count())

    trainD0, testD0 = classD0.randomSplit([0.8, 0.2], seed=42)
    trainD1, testD1 = classD1.randomSplit([0.8, 0.2], seed=42)

    trainD = trainD0.union(trainD1)
    testD = testD0.union(testD1)

    classA0 = df.filter(df["alive"]==0)
    classA1 = df.filter(df["alive"]==1)
    print("Class 0 (alive= 0): ", classA0.count())
    print("Class 1 (alive= 1): ", classA1.count())

    trainA0, testA0 = classA0.randomSplit([0.8, 0.2], seed=42)
    trainA1, testA1 = classA1.randomSplit([0.8, 0.2], seed=42)

    trainA = trainA0.union(trainA1)
    testA = testA0.union(testA1)

    rfD = RandomForestClassifier(labelCol = 'deceased', featuresCol = 'features')
    rfA = RandomForestClassifier(labelCol = 'alive', featuresCol = 'features')    
    
    
    lrModelD = rfD.fit(trainD)
    resultsD = lrModelD.transform(testD)

    lrModelA = rfA.fit(trainA)
    resultsA = lrModelA.transform(testA)

    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    metrics_bin=['areaUnderROC', 'areaUnderPR']
    print('Deceased label')
    print('Accuracy: {:0.2f}'.format(lrModelD.evaluate(testD).accuracy))
    my_evalD = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    my_eval_binD = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')

    def printPerformanceD(resultsD):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_evalD.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binD.evaluate(resultsD, {my_eval_binD.metricName: m})))
        y_test = np.array(resultsD.select("deceased").collect()).flatten()
        y_pred = np.array(resultsD.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    resultsD = lrModelD.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceD(resultsD)

    print('Alive label')
    print('Accuracy: {:0.2f}'.format(lrModelA.evaluate(testA).accuracy))
    my_evalA = MulticlassClassificationEvaluator(labelCol = 'alive', predictionCol = 'prediction')
    my_eval_binA = BinaryClassificationEvaluator(labelCol ='alive', rawPredictionCol = 'prediction')

    def printPerformanceA(resultsA):
        for m in metrics: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_evalA.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_binA.evaluate(resultsA, {my_eval_binA.metricName: m})))
        y_test = np.array(resultsA.select("deceased").collect()).flatten()
        y_pred = np.array(resultsA.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
 
    resultsA = lrModelA.transform(df)
    print("\n[Performance on Whole set]")
    printPerformanceA(resultsA)
    
    return resultsD

@transform_pandas(
    Output(rid="ri.vector.main.execute.a6e6a4d2-b6c0-4e24-86f2-236dca024c8a"),
    sample=Input(rid="ri.foundry.main.dataset.ce49c9a4-0100-4401-8015-8c4b250c56b7")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def RandomForestDem(sample):
    #parameters
    maxCategories=4
    seed=42

    df = sample
    label = 'deceased'    
    featureIndexer =VectorIndexer(inputCol="features", outputCol="indexed_features", maxCategories=maxCategories).fit(df)

    # stratified split
    y = df.select(df['deceased']).collect() #.toPandas().values.ravel()
    X = df.drop(label, "features") #.toPandas()        

    # stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    
    y_train = train.select('deceased') #.toPandas().values.ravel()
    X_train = train.drop(label, "features") #.toPandas()
    y_test = test.select(df['deceased']) #.toPandas().values.ravel()
    X_test = test.drop(label, "features") #.toPandas()
    columns = X_train.columns
    print("\n[Dataframe shape]")
    print("Train set: {}, Spark DF Shape: rows- {} columns- {}\n{}".format(train.count(), X_train.count(), len(X_train.columns), train.groupBy("deceased").count()))
    print("Test set: {}, Spark DF Shape: rows- {} columns- {}\n{}".format(test.count(), X_train.count(), len(X_train.columns), test.groupBy("deceased").count()))

    estimator = RandomForestClassifier(featuresCol='indexed_features', labelCol="deceased", seed=seed,
                                numTrees=10)
    pipeline = Pipeline(stages=[featureIndexer, estimator])
    evaluator=MulticlassClassificationEvaluator(labelCol='deceased', predictionCol='prediction', metricName='accuracy')

    model = pipeline.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    ## full list of hyperparameters
    param_list = ['maxDepth', 'impurity', 'minInstancesPerNode', 'numTrees']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxDepth, [10, 20, 30])\
            .addGrid(estimator.impurity, ['entropy', 'gini'])\
            .addGrid(estimator.minInstancesPerNode, [1, 5, 10])\
            .addGrid(estimator.numTrees, [20, 50, 100])\
            .build()

    ## short list of hyperparameters        
    # param_list = ['maxDepth', 'impurity', 'numTrees']
    # paramGrid = ParamGridBuilder()\
    #         .addGrid(estimator.maxDepth, [10, 30])\
    #         .addGrid(estimator.impurity, ['entropy', 'gini'])\
    #         .addGrid(estimator.numTrees, [20, 100])\
    #         .build()
    
    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)
    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))
    
    best_params={param.name: value for param, value in zip(model.bestModel.stages[1].extractParamMap().keys(), model.bestModel.stages[1].extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])
  
    my_eval = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("deceased").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)    
    
    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)  

    estimator.setParams(**best_params)
    model = pipeline.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)       
    
    return df_best_params

@transform_pandas(
    Output(rid="ri.vector.main.execute.a29983f5-39d1-4ebd-8b9b-ff0b47a93d34"),
    deceasedSet=Input(rid="ri.foundry.main.dataset.74e51259-2e8a-4744-95b3-2a5b7c1bdb66")
)
from pyspark.ml.classification import RandomForestClassifier
import numpy as np
def RandomForestDemo(deceasedSet):
    #parameters
    maxCategories=4
    seed=42

    df = deceasedSet
    label = 'deceased'    

    # stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1)

    rf = RandomForestClassifier(labelCol = 'deceased', featuresCol = 'features')
    evaluator=MulticlassClassificationEvaluator(labelCol='deceased', predictionCol='prediction', metricName='accuracy')
    model = rf.fit(train)
    predictions = model.transform(test)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(predictions)))
    return predictions

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.5d203428-cfc6-4527-b940-9af1c4af79cf"),
    eraPrep=Input(rid="ri.foundry.main.dataset.341c241d-8523-4742-afa6-213415ddc774")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer, VarianceThresholdSelector
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
from pyspark.sql.types import *

def RfFeatureImportance(eraPrep):
    ## Keeping only 4/9 initially
    #parameters
    maxCategories=4
    seed=42
    label="deceased"

    df = eraPrep
    estimator = RandomForestClassifier(labelCol="deceased", featuresCol="features", seed = seed, numTrees=10, impurity='gini', featureSubsetStrategy="auto", subsamplingRate = 0.8)

    model = estimator.fit(df)
    result = model.transform(df)
    importances = model.featureImportances
    indices = np.argsort(importances)

    
    featuresList = model.toDebugString#ExtractFeatureImp(model.toDebugStringfeatureImportances, df, "features")
    print(featuresList)
    #featuresIdx = [x for x in varlist['idx'][0:50]]

    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.8f7abd8e-6518-4eba-9f4e-0370ece106cf"),
    condPrep=Input(rid="ri.foundry.main.dataset.d03ab161-fabe-4425-b2e6-63474c6e6a11")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer, VarianceThresholdSelector
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
from pyspark.sql.types import *

def RfFeatureImportance2(condPrep):
    ## Keeping only 4/9 initially
    #parameters
    maxCategories=4
    seed=42
    label="deceased"

    df = condPrep
    estimator = RandomForestClassifier(labelCol="deceased", featuresCol="features", seed = seed, numTrees=10,impurity='gini', featureSubsetStrategy="auto", subsamplingRate = 0.8)

    model = estimator.fit(df)
    result = model.transform(df)
    importances = model.featureImportances
    indices = np.argsort(importances)

    
    featuresList = model.toDebugString#ExtractFeatureImp(model.toDebugStringfeatureImportances, df, "features")
    print(featuresList)
    #featuresIdx = [x for x in varlist['idx'][0:50]]

    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.7e4c8ed4-9e8e-4bb2-b609-c48e26c8c07b"),
    obs=Input(rid="ri.foundry.main.dataset.20a84953-6c28-4d22-94ac-20105a7ff051")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer, VarianceThresholdSelector
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
from pyspark.sql.types import *

def RfFeatureImportance3(obs):
    ## Keeping only 4/9 initially
    #parameters
    maxCategories=4
    seed=42
    label="deceased"

    df = obs
    estimator = RandomForestClassifier(labelCol="deceased", featuresCol="features", seed = seed, numTrees=10,impurity='gini', featureSubsetStrategy="auto", subsamplingRate = 0.8)

    model = estimator.fit(df)
    result = model.transform(df)
    importances = model.featureImportances
    indices = np.argsort(importances)

    
    featuresList = model.toDebugString#ExtractFeatureImp(model.toDebugStringfeatureImportances, df, "features")
    print(featuresList)
    #featuresIdx = [x for x in varlist['idx'][0:50]]

    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.54289a42-87ec-41b8-ab44-527eab7af1c0"),
    expoPrep=Input(rid="ri.foundry.main.dataset.acf6f7df-4896-405c-82e4-3b556f8fe6cf")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer, VarianceThresholdSelector
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
from pyspark.sql.types import *

def RfFeatureImportance4(expoPrep):
    ## Keeping only 4/9 initially
    #parameters
    maxCategories=4
    seed=42
    label="deceased"

    df = expoPrep
    estimator = RandomForestClassifier(labelCol="deceased", featuresCol="features", seed = seed, numTrees=10,impurity='gini', featureSubsetStrategy="auto", subsamplingRate = 0.8)

    model = estimator.fit(df)
    result = model.transform(df)
    importances = model.featureImportances
    indices = np.argsort(importances)

    
    featuresList = model.toDebugString#ExtractFeatureImp(model.toDebugStringfeatureImportances, df, "features")
    print(featuresList)
    #featuresIdx = [x for x in varlist['idx'][0:50]]

    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.6c657d79-ad34-4b4b-9233-892bfe75e18d"),
    drugPrep=Input(rid="ri.foundry.main.dataset.1914a0ea-151f-49bc-9981-e2ffb0ac9aae")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer, VarianceThresholdSelector
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
from pyspark.sql.types import *

def RfFeatureImportance5(drugPrep):
    ## Keeping only 4/9 initially
    #parameters
    maxCategories=4
    seed=42
    label="deceased"

    df = drugPrep
    estimator = RandomForestClassifier(labelCol="deceased", featuresCol="features", seed = seed, numTrees=10,impurity='gini', featureSubsetStrategy="auto", subsamplingRate = 0.8)

    model = estimator.fit(df)
    result = model.transform(df)
    importances = model.featureImportances
    indices = np.argsort(importances)

    
    featuresList = model.toDebugString#ExtractFeatureImp(model.toDebugStringfeatureImportances, df, "features")
    print(featuresList)
    #featuresIdx = [x for x in varlist['idx'][0:50]]

    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.33eaf595-1609-42a2-8f1d-e8e867ff11c5"),
    condPrep=Input(rid="ri.foundry.main.dataset.d03ab161-fabe-4425-b2e6-63474c6e6a11")
)
from pyspark.ml.feature import UnivariateFeatureSelector
import numpy as np
#PySpark has 5 Feature Selectors
#VectorSlicer (slicing vector to get relevant features (!=0)), RFormula, ChiSqSelector (deprecated since version 3.1.0), UnivariateFeatureSelector, VarianceThresholdSelector

#A dense vector is backed by a double array representing its entry values, while a sparse vector is backed by two parallel arrays: indices and values.
def UnivariateFeatureSelector(condPrep):
    
    df = condPrep
    selector = UnivariateFeatureSelector(featuresCol = 'features', outputCol = 'selectedFeatures', labelCol = 'deceased', selectionMode = 'numTopFeatures')
    selector.setFeatureType('categorical').setLabelType('categorical').setSelectionThreshold(50)
    model = selector.fit(df)
    result = model.transform(df)
    print("UnivariateFeatureSelector output with top %d features selected using chi-squared (chi2)"
      % selector.getSelectionThreshold())
    result = result.select('person_id', 'features', 'selectedFeatures', 'deceased')
    print("UnivariateFeatureSelector output - top 50 features selected")
    print(np.array(df.columns)[model.selectedFeatures])
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.8f64eba0-5c8a-41c6-97cd-6099e723678f"),
    eraPrep=Input(rid="ri.foundry.main.dataset.341c241d-8523-4742-afa6-213415ddc774")
)
from pyspark.ml.feature import UnivariateFeatureSelector
import numpy as np
#PySpark has 5 Feature Selectors
#VectorSlicer (slicing vector to get relevant features (!=0)), RFormula, ChiSqSelector (deprecated since version 3.1.0), UnivariateFeatureSelector, VarianceThresholdSelector

#A dense vector is backed by a double array representing its entry values, while a sparse vector is backed by two parallel arrays: indices and values.
def UnivariateFeatureSelector2(eraPrep):
    
    df = eraPrep
    selector = UnivariateFeatureSelector(featuresCol = 'features', outputCol = 'selectedFeatures', labelCol = 'deceased', selectionMode = 'numTopFeatures')
    selector.setFeatureType('categorical').setLabelType('categorical').setSelectionThreshold(27)
    model = selector.fit(df)
    result = model.transform(df)
    print("UnivariateFeatureSelector output with top %d features selected using chi-squared (chi2)"
      % selector.getSelectionThreshold())
    result = result.select('person_id', 'features', 'selectedFeatures', 'deceased')
    print("UnivariateFeatureSelector output - top 25 features selected")
    print(np.array(df.columns)[model.selectedFeatures])
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.e485a4b8-f268-48d6-b8be-0f8a2b5a1e34"),
    obs=Input(rid="ri.foundry.main.dataset.20a84953-6c28-4d22-94ac-20105a7ff051")
)
from pyspark.ml.feature import UnivariateFeatureSelector
import numpy as np
#PySpark has 5 Feature Selectors
#VectorSlicer (slicing vector to get relevant features (!=0)), RFormula, ChiSqSelector (deprecated since version 3.1.0), UnivariateFeatureSelector, VarianceThresholdSelector

#A dense vector is backed by a double array representing its entry values, while a sparse vector is backed by two parallel arrays: indices and values.
def UnivariateFeatureSelector3(obs):
    
    df = obs
    selector = UnivariateFeatureSelector(featuresCol = 'features', outputCol = 'selectedFeatures', labelCol = 'deceased', selectionMode = 'numTopFeatures')
    selector.setFeatureType('categorical').setLabelType('categorical').setSelectionThreshold(50)
    model = selector.fit(df)
    result = model.transform(df)
    print("UnivariateFeatureSelector output with top %d features selected using chi-squared (chi2)"
      % selector.getSelectionThreshold())
    result = result.select('person_id', 'features', 'selectedFeatures', 'deceased')
    print("UnivariateFeatureSelector output - top 50 features selected")
    print(np.array(df.columns)[model.selectedFeatures])
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.6b4f3634-9177-45ec-8d0f-e4991fcbaeff"),
    expoPrep=Input(rid="ri.foundry.main.dataset.acf6f7df-4896-405c-82e4-3b556f8fe6cf")
)
from pyspark.ml.feature import UnivariateFeatureSelector
import numpy as np
#PySpark has 5 Feature Selectors
#VectorSlicer (slicing vector to get relevant features (!=0)), RFormula, ChiSqSelector (deprecated since version 3.1.0), UnivariateFeatureSelector, VarianceThresholdSelector

#A dense vector is backed by a double array representing its entry values, while a sparse vector is backed by two parallel arrays: indices and values.
def UnivariateFeatureSelector4(expoPrep):
    
    df = expoPrep
    selector = UnivariateFeatureSelector(featuresCol = 'features', outputCol = 'selectedFeatures', labelCol = 'deceased', selectionMode = 'numTopFeatures')
    selector.setFeatureType('categorical').setLabelType('categorical').setSelectionThreshold(50)
    model = selector.fit(df)
    result = model.transform(df)
    print("UnivariateFeatureSelector output with top %d features selected using chi-squared (chi2)"
      % selector.getSelectionThreshold())
    result = result.select('person_id', 'features', 'selectedFeatures', 'deceased')
    print("UnivariateFeatureSelector output - top 50 features selected")
    print(np.array(df.columns)[model.selectedFeatures])
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.e19ec377-b96e-474f-ac64-e0452af613a5"),
    drugPrep=Input(rid="ri.foundry.main.dataset.1914a0ea-151f-49bc-9981-e2ffb0ac9aae")
)
from pyspark.ml.feature import UnivariateFeatureSelector
import numpy as np
#PySpark has 5 Feature Selectors
#VectorSlicer (slicing vector to get relevant features (!=0)), RFormula, ChiSqSelector (deprecated since version 3.1.0), UnivariateFeatureSelector, VarianceThresholdSelector

#A dense vector is backed by a double array representing its entry values, while a sparse vector is backed by two parallel arrays: indices and values.
def UnivariateFeatureSelector5(drugPrep):
    
    df = drugPrep
    selector = UnivariateFeatureSelector(featuresCol = 'features', outputCol = 'selectedFeatures', labelCol = 'deceased', selectionMode = 'numTopFeatures')
    selector.setFeatureType('categorical').setLabelType('categorical').setSelectionThreshold(50)
    model = selector.fit(df)
    result = model.transform(df)
    print("UnivariateFeatureSelector output with top %d features selected using chi-squared (chi2)"
      % selector.getSelectionThreshold())
    result = result.select('person_id', 'features', 'selectedFeatures', 'deceased')
    print("UnivariateFeatureSelector output - top 50 features selected")
    print(np.array(df.columns)[model.selectedFeatures])
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.5fe716ab-79db-4827-ab75-73d530eb5669"),
    eraPrep=Input(rid="ri.foundry.main.dataset.341c241d-8523-4742-afa6-213415ddc774")
)
from pyspark.ml.feature import VarianceThresholdSelector
import numpy as np
def VarianceThres(eraPrep):
    df = eraPrep
    # stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1) 
    
    selector = VarianceThresholdSelector(varianceThreshold=(0.8*(1-0.8)), outputCol = 'selectedFeatures')

    model = selector.fit(train)
    result = model.transform(test)
    #selected_var = columns[sel.get_support()]
    #print("Output: Features with variance lower than %f are removed." % selected_var)
    result = result.select('person_id', 'features', 'selectedFeatures', 'deceased')
    print("Variance Threshold output - features selected")
    print(model.selectedFeatures)
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.71235021-bc60-4cd1-9c8e-67a41e376669"),
    condPrep=Input(rid="ri.foundry.main.dataset.d03ab161-fabe-4425-b2e6-63474c6e6a11")
)
from pyspark.ml.feature import VarianceThresholdSelector
import numpy as np
def VarianceThres2(condPrep):
    df = condPrep
    # stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1) 
    
    selector = VarianceThresholdSelector(varianceThreshold=(0.8*(1-0.8)), outputCol = 'selectedFeatures')

    model = selector.fit(train)
    result = model.transform(test)
    #selected_var = columns[sel.get_support()]
    #print("Output: Features with variance lower than %f are removed." % selected_var)
    result = result.select('person_id', 'features', 'selectedFeatures', 'deceased')
    print("Variance Threshold output - features selected")
    print(model.selectedFeatures)
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.6f8c9644-6173-48c6-acdd-f623f8ff7dad"),
    obs=Input(rid="ri.foundry.main.dataset.20a84953-6c28-4d22-94ac-20105a7ff051")
)
from pyspark.ml.feature import VarianceThresholdSelector
import numpy as np
def VarianceThres3(obs):
    df = obs
    # stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1) 
    
    selector = VarianceThresholdSelector(varianceThreshold=(0.8*(1-0.8)), outputCol = 'selectedFeatures')

    model = selector.fit(train)
    result = model.transform(test)
    #selected_var = columns[sel.get_support()]
    #print("Output: Features with variance lower than %f are removed." % selected_var)
    result = result.select('person_id', 'features', 'selectedFeatures', 'deceased')
    print("Variance Threshold output - features selected")
    print(model.selectedFeatures)
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.3bc2e69b-d476-40d7-b586-3e5d797639ac"),
    expoPrep=Input(rid="ri.foundry.main.dataset.acf6f7df-4896-405c-82e4-3b556f8fe6cf")
)
from pyspark.ml.feature import VarianceThresholdSelector
import numpy as np
def VarianceThres4(expoPrep):
    df = expoPrep
    # stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1) 
    
    selector = VarianceThresholdSelector(varianceThreshold=(0.8*(1-0.8)), outputCol = 'selectedFeatures')

    model = selector.fit(train)
    result = model.transform(test)
    #selected_var = columns[sel.get_support()]
    #print("Output: Features with variance lower than %f are removed." % selected_var)
    result = result.select('person_id', 'features', 'selectedFeatures', 'deceased')
    print("Variance Threshold output - features selected")
    print(model.selectedFeatures)
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.4a05b484-5d2f-41cf-bee1-fec4dcc25542"),
    drugPrep=Input(rid="ri.foundry.main.dataset.1914a0ea-151f-49bc-9981-e2ffb0ac9aae")
)
from pyspark.ml.feature import VarianceThresholdSelector
import numpy as np
def VarianceThres5(drugPrep):
    df = drugPrep
    # stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1) 
    
    selector = VarianceThresholdSelector(varianceThreshold=(0.8*(1-0.8)), outputCol = 'selectedFeatures')

    model = selector.fit(train)
    result = model.transform(test)
    #selected_var = columns[sel.get_support()]
    #print("Output: Features with variance lower than %f are removed." % selected_var)
    result = result.select('person_id', 'features', 'selectedFeatures', 'deceased')
    print("Variance Threshold output - features selected")
    print(model.selectedFeatures)
    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.cffb04fd-c66e-44f9-b5d4-5da7dbfe3f63"),
    Person=Input(rid="ri.foundry.main.dataset.af5e5e91-6eeb-4b14-86df-18d84a5aa010")
)
def agePredictions(Person):
    df = Person.select('person_id', 'year_of_birth', 'gender_concept_name', 'race_concept_name', 'ethnicity_concept_name','is_age_90_or_older')
 #train
    df = df.withColumn('is_age_90_or_older', when(df.is_age_90_or_older == 'true' , 1).when(df.is_age_90_or_older == 'false' , 1))

    df_missing = df.where(df['year_of_birth'].isNull()).fillna(0) #test
    race = StringIndexer(inputCol = 'race_concept_name', outputCol = 'raceIndex').setHandleInvalid('skip')
    gender = StringIndexer(inputCol = 'gender_concept_name', outputCol = 'genderIndex').setHandleInvalid('skip')
    ethnicity = StringIndexer(inputCol = 'ethnicity_concept_name', outputCol = 'ethnicityIndex').setHandleInvalid('skip')
    df = df.dropna()
    encoded = Pipeline(stages=[race, gender,ethnicity]).fit(df).transform(df)
    encMissing = Pipeline(stages=[race, gender, ethnicity]).fit(df_missing).transform(df_missing)

    assembler = VectorAssembler().setInputCols(['raceIndex', 'genderIndex', 'ethnicityIndex', 'is_age_90_or_older']).setOutputCol('features')
    encoded = assembler.transform(encoded) 
    encMissing = assembler.transform(encMissing) #missing subset
    train, test = encoded.randomSplit([0.7, 0.3])

    lr = LogisticRegression(featuresCol = 'features', labelCol = 'year_of_birth', maxIter=10)
    lrModel = lr.fit(train)
    predictions = lrModel.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol="year_of_birth", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(accuracy)
    agePredictions = lrModel.transform(encMissing)
    
    return agePredictions

@transform_pandas(
    Output(rid="ri.vector.main.execute.01c4971b-a644-4e68-99ca-6785a2b5591e"),
    condPrep=Input(rid="ri.foundry.main.dataset.d03ab161-fabe-4425-b2e6-63474c6e6a11"),
    eraPrep=Input(rid="ri.foundry.main.dataset.341c241d-8523-4742-afa6-213415ddc774")
)
def compare(eraPrep, condPrep):
    df1 = eraPrep
    df2 = condPrep
    col1 = df1.columns
    col2 = df2.columns
    same = set(col1).intersection(col2)
    print(same)

@transform_pandas(
    Output(rid="ri.vector.main.execute.51a213df-97b8-46a5-91f5-0cf0676ac3a2"),
    drugPrep=Input(rid="ri.foundry.main.dataset.1914a0ea-151f-49bc-9981-e2ffb0ac9aae"),
    expoPrep=Input(rid="ri.foundry.main.dataset.acf6f7df-4896-405c-82e4-3b556f8fe6cf")
)
def compare2(expoPrep, drugPrep):
    df1 = expoPrep
    df2 = drugPrep
    col1 = df1.columns
    col2 = df2.columns
    same = set(col1).intersection(col2)
    print(same)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.51515795-2599-4e16-a4e8-e7e63f2bedb0"),
    condition_era_samp=Input(rid="ri.foundry.main.dataset.14761745-9012-429c-8933-73e0aae4dfb4")
)
def condEra(condition_era_samp):
    df = condition_era_samp
    print(df.columns)
    cols = ['person_id', 'condition_concept_id', 'condition_era_start_date', 'condition_era_end_date', 'condition_occurrence_count']
    df = df[cols]
    #df = df.na.fill(value=0,subset=["condition_era_end_date"])
    df = df.withColumn('condition_era_duration', datediff(df.condition_era_end_date, df.condition_era_start_date)).drop('condition_era_start_date', 'condition_era_end_date').fillna(0)
    
    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.6ce3fb43-0318-4ed0-a201-c7b45654c395"),
    conditions_to_macrovisits=Input(rid="ri.foundry.main.dataset.85e1ac5b-3421-4747-9ea2-50b28bd0bc65")
)
def condMac(conditions_to_macrovisits):
    df = conditions_to_macrovisits
    print(df.columns)
    cols = ['person_id', 'condition_occurrence_id', 'macrovisit_id', 'macrovisit_start_date', 'macrovisit_end_date']
    df = df[cols]

    
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.80ade8e7-23be-47cd-8e17-b2446fa2eb70"),
    condition_occurrence_samp=Input(rid="ri.foundry.main.dataset.891b4ab5-c85d-4baf-acd1-5ad4073dc09b")
)
def condOcc(condition_occurrence_samp):
    df = condition_occurrence_samp
    print(df.columns)
    cols = ['person_id','condition_concept_id', 'condition_start_date', 'condition_end_date']
    df = df[cols]
    df = df.na.fill(value=0,subset=["condition_end_date"])
    df = df.withColumn('condition_duration', datediff(df.condition_end_date, df.condition_start_date)).drop('condition_start_date', 'condition_end_date', 'condition_occurrence_id', 'visit_occurrence_id').fillna(0)
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d03ab161-fabe-4425-b2e6-63474c6e6a11"),
    condOcc=Input(rid="ri.foundry.main.dataset.80ade8e7-23be-47cd-8e17-b2446fa2eb70"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69")
)
def condPrep(condOcc, death):
    df = condOcc

    dfD = death
    deceased = dfD.select('person_id').collect()
    deceasedList = [row.person_id for row in deceased]
    df = df.groupBy('person_id').pivot('condition_concept_id').count().fillna(0)

    df = df.withColumn('deceased', when(df['person_id'].isin(deceasedList) , 1)).fillna(0) # Alive = 0, Deceased = 1
    df = df.withColumn('alive', when(~df['person_id'].isin(deceasedList) , 1)).fillna(0) # Alive = 1, Deceased = 0
    print(df.columns)
    
    #conditions = df.select('condition_concept_id').distinct().rdd.flatMap(lambda x : x).collect()
    #conditions.sort()
    #print(conditions.count())

    cols = list(set(df.columns) - {'person_id', 'deceased','alive'})
    assembler = VectorAssembler().setInputCols(cols).setOutputCol('features')
    df = assembler.transform(df) 
    patientCount = df.select('person_id').distinct().count()
    print(patientCount)
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.74e51259-2e8a-4744-95b3-2a5b7c1bdb66"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    demPrepare=Input(rid="ri.foundry.main.dataset.993a59f6-65d0-43f8-abd6-7bf7e232dc0f")
)
def deceasedSet(death, demPrepare):
    dfD = death
    deceased = dfD.select('person_id').collect()
    deceasedList = [row.person_id for row in deceased]
    
    df = demPrepare
    df = df.withColumn('deceased', when(df['person_id'].isin(deceasedList) , 1)).fillna(0)
    df = df.withColumn('alive', when(~df['person_id'].isin(deceasedList) , 0)).fillna(1)
    cols = ['is_age_90_or_older', 'age', 'gender_fem' , 'gender_mal', 'gender_unk', 'ethnicity_his', 'race_whi', 'race_bla', 'race_mult', 'race_unk', 'race_asi', 'race_nat', 'race_ind', 'ethnicity_unk', 'ethnicity_notHis', 'ageGroup_infant', 'ageGroup_toddler', 'ageGroup_adolescent', 'ageGroup_youngAd', 'ageGroup_adult', 'ageGroup_olderAd', 'ageGroup_elderly']
    assembler = VectorAssembler().setInputCols(cols).setOutputCol('features')
    df = assembler.transform(df) 
    patientCount = df.select('person_id').distinct().count()
    print(patientCount)
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.993a59f6-65d0-43f8-abd6-7bf7e232dc0f"),
    mergingAge=Input(rid="ri.foundry.main.dataset.30eb0dfa-e6ad-47ca-8683-52ef2abb1159")
)
def demPrepare(mergingAge):
    df = mergingAge
    gender = df.select('gender_concept_name').distinct().collect()
    race = df.select('race_concept_name').distinct().collect()
    ethnicity = df.select('ethnicity_concept_name').distinct().collect()
    print('Gender count', gender)
    print('Race count', race)
    print('Ethnicity count', ethnicity)

    #gender columns
    other = ['Unknown', 'UNKNOWN', 'No matching concept', 'OTHER', 'Gender unknown']
    df = df.withColumn('gender_fem', when(col('gender_concept_name') == 'FEMALE', 1)).withColumn('gender_mal', when(col('gender_concept_name') == 'MALE', 1)).withColumn('gender_unk', when(col('gender_concept_name').isin(other), 1)).fillna(0)

    #race columns
    black = ['Barbadian', 'African American', 'Haitian', 'Melanesian', 'Micronesian', 'Jamaican', 'Black' , 'Trinidadian', 'Black or African American', 'Madagascar']
    multi = ['Multiple race', 'Multiple races', 'More than one race']
    unk = [None, 'Other Race', 'Unknown racial group', 'no information', 'Unknown racial group', 'Refuse to answer', 'No matching concept']
    asi = ['Thai', 'Chinese', 'Indonesian', 'Taiwanese', 'Japanese', 'Filipino', 'Vietnamese', 'Pakistani', 'Singaporean', 'Laotian', 'Korean', 'Malaysian', 'Okinawan', 'Asian Indian', 'Asian or Pacific Islander', 'Nepalese', 'Bhutanese', 'Burmese', 'Cambodian', 'Hmong', 'Asian Indian']
    nat = ['Other Pacific Islander', 'Dominica Islander', 'Polynesian', 'Native Hawaiian or Other Pacific Islander']
    ind = ['Sri Lankan', 'West Indian', 'Bangladeshi', 'American Indian or Alaska Native','Maldivian']

    df = df.withColumn('ethnicity_his', when(col('race_concept_name') == 'Hispanic', 1)).withColumn('race_whi', when(col('race_concept_name') == 'White', 1)).withColumn('race_bla', when(col('race_concept_name').isin(black), 1)).withColumn('race_mult', when(col('race_concept_name').isin(multi), 1)).withColumn('race_unk', when(col('race_concept_name').isin(unk), 1)).withColumn('race_asi', when(col('race_concept_name').isin(asi), 1)).withColumn('race_nat', when(col('race_concept_name').isin(nat), 1)).withColumn('race_ind', when(col('race_concept_name').isin(ind), 1)).fillna(0)

    #ethnicity columns
    nh = ['No information', 'Unknown', 'Other', 'Refuse to answer', 'No matching concept', 'Other/Unknown', 'Patient ethnicity unknown',None]
    df = df.withColumn('ethnicity_unk', when(col('ethnicity_concept_name').isin(nh), 1)).withColumn('ethnicity_his', when(col('ethnicity_concept_name') == 'Hispanic or Latino', 1)).withColumn('ethnicity_notHis', when(col('ethnicity_concept_name') == 'Not Hispanic or Latino', 1)).fillna(0)

    df = df.withColumn('ageGroup_infant', when(col('age') < 2, 1)).withColumn('ageGroup_toddler', when(((col('age') >= 2) & (col('age') < 4)), 1)).withColumn('ageGroup_adolescent', when(((col('age') >= 4) & (col('age') < 14)), 1)).withColumn('ageGroup_youngAd', when(((col('age') >= 14) & (col('age') < 30)), 1)).withColumn('ageGroup_adult', when(((col('age') >= 30) & (col('age') < 50)), 1)).withColumn('ageGroup_olderAd', when(((col('age') >= 50) & (col('age') < 90)), 1)).withColumn('ageGroup_elderly', when((col('is_age_90_or_older') == 'true'), 1)).fillna(0)

    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.1847123b-4884-4f57-aed1-c584d6a086b8"),
    device_exposure=Input(rid="ri.foundry.main.dataset.0d7eb4aa-5991-49de-9915-489d7184720c")
)
def devices(device_exposure):
    df = device_exposure
    print(df.columns)
    cols =['person_id', 'device_exposure_id', 'device_concept_id', 'device_exposure_start_date','device_exposure_end_date', 'device_type_concept_id', 'visit_occurrence_id', 'device_concept_name']
    df = df[cols]
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.dbc7ed10-0dd1-4093-bb8c-93f6d5ee6cc9"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    drug_era_samp=Input(rid="ri.foundry.main.dataset.84031880-1a8b-4e84-9a1e-ac08e54f11b8")
)
def drug(drug_era_samp, death):
    df = drug_era_samp['person_id','drug_concept_id', 'drug_era_start_date','drug_era_end_date']
    print(df.columns)
    df = df.na.fill(value=0,subset=["drug_era_end_date"])
    df = df.withColumn('drug_era_duration', datediff(df.drug_era_end_date, df.drug_era_start_date)).drop('drug_era_start_date', 'drug_era_end_date')

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.1914a0ea-151f-49bc-9981-e2ffb0ac9aae"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    drug=Input(rid="ri.foundry.main.dataset.dbc7ed10-0dd1-4093-bb8c-93f6d5ee6cc9")
)
def drugPrep(drug, death):
    df = drug
    sam = df.sample(withReplacement=False, fraction=0.2)
    
    dfD = death
    deceased = dfD.select('person_id').collect()
    deceasedList = [row.person_id for row in deceased]

    df = df.withColumn('deceased', when(df['person_id'].isin(deceasedList) , 1)).fillna(0) # Alive = 0, Deceased = 1
    df = df.withColumn('alive', when(~df['person_id'].isin(deceasedList) , 1)).fillna(0) # Alive = 1, Deceased = 0
    print(df.columns)
    #assembler = VectorAssembler().setInputCols(cols).setOutputCol('features')
    #df = assembler.transform(df) 
    patientCount = df.select('person_id').distinct().count()
    print(patientCount)
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.341c241d-8523-4742-afa6-213415ddc774"),
    condEra=Input(rid="ri.foundry.main.dataset.51515795-2599-4e16-a4e8-e7e63f2bedb0"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69")
)

def eraPrep(condEra, death):
    df = condEra

    dfD = death
    deceased = dfD.select('person_id').collect()
    deceasedList = [row.person_id for row in deceased]
    df = df.groupBy('person_id').pivot('condition_concept_id').count().fillna(0)

    df = df.withColumn('deceased', when(df['person_id'].isin(deceasedList) , 1)).fillna(0) # Alive = 0, Deceased = 1
    df = df.withColumn('alive', when(~df['person_id'].isin(deceasedList) , 1)).fillna(0) # Alive = 1, Deceased = 0
    #print(df.dtypes)
    
    #conditions = df.select('condition_concept_id').distinct().rdd.flatMap(lambda x : x).collect()
    #conditions.sort()
    #print(conditions.count())

    cols = list(set(df.columns) - {'person_id', 'deceased','alive'})
    assembler = VectorAssembler().setInputCols(cols).setOutputCol('features')
    df = assembler.transform(df) 
    patientCount = df.select('person_id').distinct().count()
    print(patientCount)
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8dfc3efe-0b2c-413e-9d66-ff64fa6634b0"),
    drug_exposure_samp=Input(rid="ri.foundry.main.dataset.14340f80-fb0e-4c1a-ae82-c68374f501ef")
)
def expo(drug_exposure_samp):
    df = drug_exposure_samp
    print(df.columns)
    cols = ['person_id', 'drug_concept_id', 'drug_exposure_start_date', 'drug_exposure_end_date']
    df = df[cols]
    #df = df.na.fill(value=0,subset=["condition_era_end_date"])
    df = df.withColumn('drug_exposure_duration', datediff(df.drug_exposure_end_date, df.drug_exposure_start_date)).drop('drug_exposure_start_date', 'drug_exposure_end_date').fillna(0)
    
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.acf6f7df-4896-405c-82e4-3b556f8fe6cf"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    expo=Input(rid="ri.foundry.main.dataset.8dfc3efe-0b2c-413e-9d66-ff64fa6634b0")
)
def expoPrep(expo, death):
    df = expo
    sam = df.sample(withReplacement=False, fraction=0.2)
    
    dfD = death
    deceased = dfD.select('person_id').collect()
    deceasedList = [row.person_id for row in deceased]

    df = df.withColumn('deceased', when(df['person_id'].isin(deceasedList) , 1)).fillna(0) # Alive = 0, Deceased = 1
    df = df.withColumn('alive', when(~df['person_id'].isin(deceasedList) , 1)).fillna(0) # Alive = 1, Deceased = 0
    print(df.columns)

    patientCount = df.select('person_id').distinct().count()
    print(patientCount)

    cols = list(set(df.columns) - {'person_id', 'deceased', 'alive'})
    assembler = VectorAssembler().setInputCols(cols).setOutputCol('features')
    df = assembler.transform(df) 
    
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2a0fd95e-9aa7-49de-8237-b035b92ea4d5"),
    conditions=Input(rid="ri.foundry.main.dataset.0f6ba549-dffc-4231-b372-bdb56c4aea6d")
)
from pyspark.sql.types import StructType, StructField, IntegerType
def fullConditions(conditions):
    cond = conditions
    #conditions selected by both eraPrep and condPrep
    condList = ['45877319', '442752', '35207702', '432870', '75909', '40479192', '77670', '313459', '372448', '4174262', '37311061', '45558455', '255573', '374375', '4209423', '4180628', '440029', '377091', '4169095', '4223659', '257011', '4346975', '45573007', '435796', '4144111', '437833', '444070', '316139', '321052', '198803', '433316', '4154290', '438398', '439297', '320128', '441408', '4229440', '35211275', '436230', '200219', '437113', '442077', '195867', '436096', '194133', '436070', '45562840', '255848', '37016349', '318736', '4195085', '35207062', '27674', '319835', '35207098', '765131', '75860', '762294', '80180', '78232', '37017432', '434613', '4236484', '4193704', '312437', '197684', '196523', '439777', '40405599', '376208', '201618', '4059290', '314658', '376065', '438720', '136788', '437827', '437663', '37201113', '35211350', '438485', '201620', '197988', '35211263', '4147829', '45766714', '4041283', '313217', '35207800', '31967', '201826', '80502', '443211', '37018196', '134736', '4273307', '254761', '437677', '437246', 'person_id', '200843', '35211400', '140673', '46271022', '138384', '4272240', '378253', '140821', '45768910', '45602003', '444101', '197381', '320536', '378427', '44782429', '44784217', '197320', '440704', '4079750', '601622', '437390', '314666', '4134121', '4170554', '443597', '444094', '442588', '440383', '436659', '35209011', '40479576', '434005', '4282096', '4146581', '35207170', '24134', '35208969', '442793', '45534429', '45558454', '45538868', '35207784', '4171917', '81902', '4043042', '141932', '443769', '140214', '764123', '45557144', '435515', '257007', '35211388', '4077577', '4329041', '436962', '45562109', '4001450', '35211336', '433736', '4214376', '31317', '42537748', '25297', '432867', '601619', '77074', '315078', '439407', '43531578']

    mySchema = StructType([StructField(c, IntegerType()) for c in condList])
    df = spark.createDataFrame(data=[], schema=mySchema)
    df = cond.join(df, how='left')

    for c in condList:
        df = df.withColumn(c, when(col('condition_concept_id') == c, 1)).fillna(0)

    return df
    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.8aacdc92-e794-400c-9c12-deab71d002dd"),
    drugs=Input(rid="ri.foundry.main.dataset.0672e6e0-04bc-44c2-9027-c32f022a9996")
)
from pyspark.sql.types import StructType, StructField, IntegerType
def fullDrugs(drugs):
    drug = drugs
    colsList = ['708298', '920293', '19127213', '1112921', '948078', '19079524', '19077884', '1367571', '19135374', '40167259', '1550557', '1713332', '1741122', '985247', '1301152', '40223757', '46275993', '750982', '40048828', '1322184', '1549786', '993631', '752061', '35604506', '1550023', '989878', '19035704', '915175', '1301025', '37003436', '19081224', '1343916', '1797513', '739138', '40221382', '19020053', '19115197', '19005968', '1149196', '42707627', '1154343', '1177480', '40076609', '732893', '40163897',  '19019418', '938268', '956874', '42479436', '1346823', '1545996', '40162515', '1790812', '1361711', '903963', '37003518', '1307046', '941258', '1777806', '975125', '992956', '723013', '923645', '40064205', '1124300', '970250', '997881', '1503297', '1738521', '1332419', '1153013', '40232454', '19092849', '1707687', '46287345', '1000560', '939506', '1707164', '40232756', '1154029', '1501700', '991710', '19075034', '797399', '1748975', '1112807', '1396131', '40221385', '901656', '19134047', '966991', '40160973', '1125315', '46275999', '1129625', '40220357', '1771162', '922570', '1518254', '1154186', '19127784', '1143374', '1367500', '778474', '1149380', '1201620', '36250141', '953076', '46287338', '1136980', '777221', '35605480', '715259', '40173507', '1754994', '19095164', '939259', '1596977', '715939', '19080985', '1518606', '791967', '1521369', '906780', '1560524', '1502905', '924566', '19111620', '1734104', '1127433', '1746114', '939976', '19070869', '924939', '19078924', '35603428', '40240688', '19019073', '967823', '1551170', '19005965', '781039', '35605482', '1135766', '19049105', '755695', '46275996', '951511', '19027362', '19077548', '19080217', '1332418', '1174888', '986417', '1107830', '1154161', '1115008', '43013024', '40227049', '40173508', '40221384', '40227012', '974166', '766529', '1373928', '718583', '19036781', '40220386', '19003953', '1344143', '1124957', '1308738', '1705674', '1510813', '836208', '1103314', '1786621', '1759206', '948582', '1386957', '19045045', '19137312', '1836430', '1110410', '40221381', '19127890', '785649', '753626', '1551099', '1759842', '1150345', '1126658', '715233', '46287424', '19011773', '19037038', '1308216', '778711', '1136601', '1545958', '40241504', '703547', '1114220', '19093848', '19061088', '19011035', '1163944', '1506270']
    mySchema = StructType([StructField(c, IntegerType()) for c in colsList])
    df = spark.createDataFrame(data=[], schema=mySchema)
    df = drug.join(df, how='left')

    for c in colsList:
        df = df.withColumn(c, when(col('drug_concept_id') == c, 1)).fillna(0)

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.7491d632-f8cf-46cc-a2d1-f828f153f006"),
    obs=Input(rid="ri.foundry.main.dataset.20a84953-6c28-4d22-94ac-20105a7ff051"),
    observation=Input(rid="ri.foundry.main.dataset.6ed38f3b-3fc9-4131-89d9-e443bb8c54fc")
)
from pyspark.sql.types import StructType, StructField, IntegerType

def fullObservations(observation, obs):
    sample = obs
    colsList = list(set(sample.columns) - {'person_id', 'deceased', 'features'})

    raw = observation

    mySchema = StructType([StructField(c, IntegerType()) for c in colsList])
    df = spark.createDataFrame(data=[], schema=mySchema)
    df = raw.join(df, how='left')

    for c in colsList:
        df = df.withColumn(c, when(col('observation_concept_id') == c, 1)).fillna(0)

    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.ea50730f-6d66-4eee-b7c9-790b6fa9b214"),
    eraPrep=Input(rid="ri.foundry.main.dataset.341c241d-8523-4742-afa6-213415ddc774")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def logisticCon1(eraPrep):
    #parameters
    seed=42

    df = eraPrep  
# stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1)    
    ## if regParam=0 -> Ridge regression, elasticNetParam=1 -> Lasso
    estimator = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.1).setLabelCol('deceased')
    
    model = estimator.fit(train)
    print(model.coefficients)
    print(model.intercept)
    result = model.evaluate(test)
    
    print('FPR: ', result.falsePositiveRateByLabel)
    print('TPR: ', result.truePositiveRateByLabel)
    print('Weighted FPR: ', result.weightedFalsePositiveRate)
    print('Weighted TPR: ', result.weightedTruePositiveRate)
    print('Weighted Precision: ', result.weightedPrecision)
    print('Weighted Recall: ', result.weightedRecall)

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.2cb2bc2f-827b-4db1-81f2-779229abc095"),
    condPrep=Input(rid="ri.foundry.main.dataset.d03ab161-fabe-4425-b2e6-63474c6e6a11")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def logisticCon2(condPrep):
    #parameters
    seed=42

    df = condPrep
  
# stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1)    
    ## if regParam=0 -> Ridge regression, elasticNetParam=1 -> Lasso
    estimator = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.1).setLabelCol('deceased')
    
    model = estimator.fit(train)
    print(model.coefficients)
    print(model.intercept)
    result = model.evaluate(test)
    
    print('FPR: ', result.falsePositiveRateByLabel)
    print('TPR: ', result.truePositiveRateByLabel)
    print('Weighted FPR: ', result.weightedFalsePositiveRate)
    print('Weighted TPR: ', result.weightedTruePositiveRate)
    print('Weighted Precision: ', result.weightedPrecision)
    print('Weighted Recall: ', result.weightedRecall)

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.84fea510-fdfb-488d-86fc-84c2933cbe1b"),
    deceasedSet=Input(rid="ri.foundry.main.dataset.74e51259-2e8a-4744-95b3-2a5b7c1bdb66")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def logisticDem(deceasedSet):
    #parameters
    seed=42

    df = deceasedSet  
    ## stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    print("[Dataframe shape]")
    print("Train set: {}\n {}".format(train.count(), train.groupBy("deceased").count().toPandas()))
    print("Test set: {}\n {}".format(test.count(), test.groupBy("deceased").count().toPandas()))
    
    ## if regParam=0 -> Ridge regression, elasticNetParam=1 -> Lasso
    estimator = LogisticRegression(featuresCol='features', labelCol='deceased')
    evaluator=MulticlassClassificationEvaluator(labelCol='deceased', predictionCol='prediction', metricName='accuracy')
    
    model = estimator.fit(train)
    print('Default parameters accuracy on test set: {:f}'.format(evaluator.evaluate(model.transform(test))))

    param_list= ['maxIter', 'regParam', 'elasticNetParam']
    paramGrid = ParamGridBuilder()\
            .addGrid(estimator.maxIter, [10, 50, 100, 500, 1000])\
            .addGrid(estimator.regParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .addGrid(estimator.elasticNetParam, [0.0, 0.001, 0.1, 0.5, 1])\
            .build()

    tvs = TrainValidationSplit(estimator=estimator,
                           estimatorParamMaps=paramGrid,
                           evaluator=evaluator,
                           trainRatio=0.8,
                           seed=seed,
                           parallelism=10)                   

    model = tvs.fit(train)
    results = model.transform(test)
    print('Best model accuracy on test set: {:f}'.format(evaluator.evaluate(results)))
    
    best_params={param.name: value for param, value in zip(model.bestModel.extractParamMap().keys(), model.bestModel.extractParamMap().values()) if param.name in param_list}
    print("\n[Best hyperparameters]")
    for p in param_list: print('Best {}: {}'.format(p, best_params[p]))
    print("\n")
    df_best_params = pd.DataFrame(data=best_params, index=[0])
  
    my_eval = MulticlassClassificationEvaluator(labelCol = 'deceased', predictionCol = 'prediction')
    metrics=['accuracy', 'weightedPrecision', 'weightedRecall', 'f1']
    my_eval_bin = BinaryClassificationEvaluator(labelCol ='deceased', rawPredictionCol = 'prediction')
    metrics_bin=['areaUnderROC', 'areaUnderPR']

    def printPerformance(results):
        for m in metrics: print('{}: {:f}'.format(m, my_eval.evaluate(results, {my_eval.metricName: m})))
        for m in metrics_bin: print('{}: {:f}'.format(m, my_eval_bin.evaluate(results, {my_eval_bin.metricName: m})))
        y_test = np.array(results.select("deceased").collect()).flatten()
        y_pred = np.array(results.select("prediction").collect()).flatten()
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("[Performance on Test set]")
    printPerformance(results)    
    
    results = model.transform(df)
    print("\n[Performance on Whole set]")
    printPerformance(results)

    estimator.setParams(**best_params)
    model = estimator.fit(df)
    results = model.transform(df)
    print("\n[Performance on Whole set with Fitting]")
    printPerformance(results)   

    return df_best_params

@transform_pandas(
    Output(rid="ri.vector.main.execute.c89cb9c7-7c8e-4c5b-9996-23a092edb2e5"),
    expoPrep=Input(rid="ri.foundry.main.dataset.acf6f7df-4896-405c-82e4-3b556f8fe6cf")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def logisticDrug1(expoPrep):
    #parameters
    seed=42

    df = expoPrep
  
# stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1)    
    ## if regParam=0 -> Ridge regression, elasticNetParam=1 -> Lasso
    estimator = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.1).setLabelCol('deceased')
    
    model = estimator.fit(train)
    print(model.coefficients)
    print(model.intercept)
    result = model.evaluate(test)
    
    print('FPR: ', result.falsePositiveRateByLabel)
    print('TPR: ', result.truePositiveRateByLabel)
    print('Weighted FPR: ', result.weightedFalsePositiveRate)
    print('Weighted TPR: ', result.weightedTruePositiveRate)
    print('Weighted Precision: ', result.weightedPrecision)
    print('Weighted Recall: ', result.weightedRecall)

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.55efc92a-d0e1-4969-97cb-6373712b4d32"),
    drugPrep=Input(rid="ri.foundry.main.dataset.1914a0ea-151f-49bc-9981-e2ffb0ac9aae")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def logisticDrug2(drugPrep):
    #parameters
    seed=42

    df = drugPrep
  
# stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1)    
    ## if regParam=0 -> Ridge regression, elasticNetParam=1 -> Lasso
    estimator = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.1).setLabelCol('deceased')
    
    model = estimator.fit(train)
    print(model.coefficients)
    print(model.intercept)
    result = model.evaluate(test)
    
    print('FPR: ', result.falsePositiveRateByLabel)
    print('TPR: ', result.truePositiveRateByLabel)
    print('Weighted FPR: ', result.weightedFalsePositiveRate)
    print('Weighted TPR: ', result.weightedTruePositiveRate)
    print('Weighted Precision: ', result.weightedPrecision)
    print('Weighted Recall: ', result.weightedRecall)

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.73258be3-d7dc-4398-9490-a12f16a83e5e"),
    obs=Input(rid="ri.foundry.main.dataset.20a84953-6c28-4d22-94ac-20105a7ff051")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix

def logisticObs(obs):
    #parameters
    seed=42

    df = obs
  
# stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=42)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=42)

    train = train0.union(train1)
    test = test0.union(test1)    
    ## if regParam=0 -> Ridge regression, elasticNetParam=1 -> Lasso
    estimator = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.1).setLabelCol('deceased')
    
    model = estimator.fit(train)
    print(model.coefficients)
    print(model.intercept)
    result = model.evaluate(test)
    
    print('FPR: ', result.falsePositiveRateByLabel)
    print('TPR: ', result.truePositiveRateByLabel)
    print('Weighted FPR: ', result.weightedFalsePositiveRate)
    print('Weighted TPR: ', result.weightedTruePositiveRate)
    print('Weighted Precision: ', result.weightedPrecision)
    print('Weighted Recall: ', result.weightedRecall)

    

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.30eb0dfa-e6ad-47ca-8683-52ef2abb1159"),
    Person=Input(rid="ri.foundry.main.dataset.af5e5e91-6eeb-4b14-86df-18d84a5aa010"),
    agePredictions=Input(rid="ri.foundry.main.dataset.cffb04fd-c66e-44f9-b5d4-5da7dbfe3f63")
)
def mergingAge(agePredictions, Person):
    missing = agePredictions.select('person_id', 'year_of_birth', 'gender_concept_name', 'race_concept_name', 'ethnicity_concept_name','is_age_90_or_older', 'prediction') #missing subset
    
    df = Person.select('person_id', 'year_of_birth', 'gender_concept_name', 'race_concept_name', 'ethnicity_concept_name','is_age_90_or_older').dropna()
    df = df.withColumn('is_age_90_or_older', when(df.is_age_90_or_older == 'true' , 1).when(df.is_age_90_or_older == 'false' , 1))
    race = StringIndexer(inputCol = 'race_concept_name', outputCol = 'raceIndex').setHandleInvalid('skip')
    gender = StringIndexer(inputCol = 'gender_concept_name', outputCol = 'genderIndex').setHandleInvalid('skip')
    ethnicity = StringIndexer(inputCol = 'ethnicity_concept_name', outputCol = 'ethnicityIndex').setHandleInvalid('skip')
    df = df.dropna()
    encoded = Pipeline(stages=[race, gender,ethnicity]).fit(df).transform(df)
    assembler = VectorAssembler().setInputCols(['raceIndex', 'genderIndex', 'ethnicityIndex', 'is_age_90_or_older']).setOutputCol('features')
    encoded = assembler.transform(encoded) 

    encoded = encoded.withColumn('age', (2021 - col('year_of_birth'))).drop('raceIndex', 'genderIndex', 'ethnicityIndex', 'features')
    missing = missing.withColumn('year_of_birth', col('prediction'))
    missing = missing.withColumn('age', (2021 - col('year_of_birth'))).drop('prediction')
    df_ = encoded.union(missing).drop_duplicates()
    return df_

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.20a84953-6c28-4d22-94ac-20105a7ff051"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    observation_samp=Input(rid="ri.foundry.main.dataset.66b44919-bc6a-4d5f-a019-2dc8216cf5ab")
)
def obs(observation_samp, death):
    df = observation_samp['person_id','observation_concept_id']

    dfD = death
    deceased = dfD.select('person_id').collect()
    deceasedList = [row.person_id for row in deceased]
    df = df.groupBy('person_id').pivot('observation_concept_id').count().fillna(0)

    df = df.withColumn('deceased', when(df['person_id'].isin(deceasedList) , 1)).fillna(0) # Alive = 0, Deceased = 1
    df = df.withColumn('alive', when(~df['person_id'].isin(deceasedList) , 1)).fillna(0) # Alive = 1, Deceased = 0
    print(df.dtypes)
    
    #conditions = df.select('condition_concept_id').distinct().rdd.flatMap(lambda x : x).collect()
    #conditions.sort()
    #print(conditions.count())

    cols = list(set(df.columns) - {'person_id', 'deceased','alive'})
    assembler = VectorAssembler().setInputCols(cols).setOutputCol('features')
    df = assembler.transform(df) 
    patientCount = df.select('person_id').distinct().count()
    print(patientCount)
    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.a9c87a83-e67d-431e-a2ed-34b3cb402a03"),
    observation_period=Input(rid="ri.foundry.main.dataset.d7ca160f-61d6-4557-97be-6d3821bbe49c")
)
def obsPeriod(observation_period):
    df_per = observation_period['person_id', 'observation_period_id', 'observation_period_start_date', 'observation_period_end_date', 'period_type_concept_id', 'period_type_concept_name']
    diff = datediff('observation_period_end_date', 'observation_period_start_date')
    df = df_per.withColumn('observation_duration', diff)
    df = df.drop('observation_period_end_date', 'observation_period_start_date')
    return df

    

@transform_pandas(
    Output(rid="ri.vector.main.execute.764b14d0-1655-45c9-937f-f3c3eab8443e"),
    eraPrep=Input(rid="ri.foundry.main.dataset.341c241d-8523-4742-afa6-213415ddc774")
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorIndexer, VarianceThresholdSelector
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder, TrainValidationSplitModel, CrossValidator
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType

def org(eraPrep):
    #parameters
    maxCategories=4
    seed=42
    label="deceased"

    df = eraPrep
    y = df.select(df['deceased']).collect() #.toPandas().values.ravel()
    X = df.drop(label, "features") #.toPandas()        

    # stratified split
    class0 = df.filter(df["deceased"]==0)
    class1 = df.filter(df["deceased"]==1)
    print("Class 0 (deceased= 0): ", class0.count())
    print("Class 1 (deceased= 1): ", class1.count())

    train0, test0 = class0.randomSplit([0.8, 0.2], seed=seed)
    train1, test1 = class1.randomSplit([0.8, 0.2], seed=seed)

    train = train0.union(train1)
    test = test0.union(test1)
    
    y_train = train.select('deceased') #.toPandas().values.ravel()
    X_train = train.drop(label, "features") #.toPandas()
    y_test = test.select(df['deceased']) #.toPandas().values.ravel()
    X_test = test.drop(label, "features") #.toPandas()
    columns = X_train.columns
    print("\n[Dataframe shape]")
    print("Train set: {}, Spark DF Shape: rows- {} columns- {}\n{}".format(train.count(), X_train.count(), len(X_train.columns), train.groupBy("deceased").count()))
    print("Test set: {}, Spark DF Shape: rows- {} columns- {}\n{}".format(test.count(), X_train.count(), len(X_train.columns), test.groupBy("deceased").count()))

    results = spark.createDataFrame([],StructType([]))
    
    # def variance_threshold(X): 
    #         from sklearn.feature_selection import VarianceThreshold

    #         # dropping columns where 1-threshold of the values are similar
    #         # a feature contains only 0s 80% of the time or only 1s 80% of the time
    #         sel = VarianceThreshold(threshold=.8*(1-.8))

    #         sel.fit_transform(X)
    #         selected_var = columns[sel.get_support()]

    #         result = spark.createDataFrame({'feature': selected_var.to_list(),
    #                             'methods': 'Variance Threshold',
    #                             'model': 'n/a',
    #                             'importance': 1})
    #         print("\nMethod: Variance Threshold")                    
    #         print('n_features_selected:',selected_var.shape[0])
    #         # print('Features Selected: ', selected_var)
    #         return result
    
    # def lasso(X, y): 
    #     lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=seed).fit(X, y)

    #     print("\nMethod: Lasso regularization")
    #     print('Train Accuracy: {:0.2f}'.format(lasso.score(X, y)))
        
    #     df_lasso = pd.DataFrame()

    #     for c, cla in zip(lasso.coef_, range(-2,3,1)):
    #         temp = pd.DataFrame({'feature': columns, 'coef': c, 'class': cla})
    #         df_lasso = pd.concat([df_lasso, temp], ignore_index=True)

    #     df_lasso2 = df_lasso.groupby(['feature'], as_index=False).agg({'coef': 'sum'})
    #     df_lasso2['Model'] = 'Lasso'

    #     df_lasso3 = df_lasso2[df_lasso2['coef']!=0].copy()
    #     df_lasso3.loc[:,'importances'] = 1
    #     result = pd.DataFrame({'feature': df_lasso3['feature'], 
    #                             'methods': 'Regularization',
    #                             'model': 'Lasso', 
    #                             'importance': df_lasso3['importances']})
    #     return result

    def RF_Imp(X_train, y_train, X_test, y_test):
        estimator = RandomForestClassifier(random_state=seed, n_jobs=-1)
        estimator.fit(X_train, y_train)
        print("\nMethod: Random Forest Feature Importance")
        print("Train Score: {:0.2f}".format(estimator.score(X_train, y_train)))
        print("Test Score: {:0.2f}".format(estimator.score(X_test, y_test)))

        result = pd.DataFrame({'feature': columns, 
                                'methods': 'Feature Importance',
                                'model': 'Random Forest', 
                                'importance': estimator.feature_importances_})
        return result

    def PMI(X_train, y_train, X_test, y_test): 
        from sklearn.inspection import permutation_importance        

        result = pd.DataFrame()
        rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
        ridge = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=seed)

        def running(estimator, model):
            estimator.fit(X_train, y_train)
            print("\nMethod: Permutation Importance with",model)
            print("Train Score: {:0.2f}".format(estimator.score(X_train, y_train)))
            print("Test Score: {:0.2f}".format(estimator.score(X_test, y_test)))
            pmi = permutation_importance(estimator, X=X_test, y=y_test, scoring='accuracy', n_jobs=-1, random_state=seed)
            temp = pd.DataFrame({'feature': columns, 
                                'methods': 'Permutation Importance',
                                'model': model, 
                                'importance': pmi['importances_mean']})
            return temp
        
        result = pd.concat([result, running(rf, "Random Forest")], ignore_index=True)
        result = pd.concat([result, running(ridge, "Ridge")], ignore_index=True)

        return result

    def RFE(X_train, y_train, X_test, y_test):
        from sklearn.feature_selection import RFECV

        result = pd.DataFrame()

        rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
        ridge = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=seed)

        def running(estimator, model):
            estimator.fit(X_train, y_train)
            print("\nMethod: Recursive Feature Elimination with",model)
            print("Train Score: {:0.2f}".format(estimator.score(X_train, y_train)))
            print("Test Score: {:0.2f}".format(estimator.score(X_test, y_test)))
            rfe = RFECV(estimator, step=1, cv=5, n_jobs=-1)
            sel = rfe.fit(X_train, y_train)
            support = sel.support_
            print('n Selected:', sel.n_features_)
            temp = pd.DataFrame({'feature': columns, 
                                    'methods': 'RFE',
                                    'model': model,
                                    'importance': [1 if ft in columns[support] else 0 for ft in columns]})
            return temp
        
        result = pd.concat([result, running(rf, "Random Forest")], ignore_index=True)
        result = pd.concat([result, running(ridge, "Ridge")], ignore_index=True)
        
        return result

    def SFS (X_train, y_train, X_test, y_test):
        from sklearn.feature_selection import SequentialFeatureSelector
        
        result = pd.DataFrame()

        # KNN = KNeighborsClassifier(n_neighbors=3)
        rf = RandomForestClassifier(random_state=seed)
        ridge = LogisticRegression(penalty='l2', solver='liblinear', C=0.1, random_state=seed)

    def running(estimator, model):
        estimator.fit(X_train, y_train)
        print("\nMethod: Sequential Feature Selecttion with", model)
        print("Train Score: {:0.2f}".format(estimator.score(X_train, y_train)))
        print("Test Score: {:0.2f}".format(estimator.score(X_test, y_test)))

        sfs = SequentialFeatureSelector(estimator, n_jobs=-1, n_features_to_select=0.5)
        sfs.fit(X_train, y_train)
        support = sfs.get_support()
        print('n Selected:', sfs.n_features_to_select_)

        temp = pd.DataFrame({'feature': columns, 
                            'methods': 'SFS',
                            'model': model, 
                            'importance': [1 if ft in columns[support] else 0 for ft in columns]})
        return temp
        
        # result = pd.concat([result, running(KNN, "KNN")], ignore_index=True)
        result = pd.concat([result, running(rf, "Random Forest")], ignore_index=True)
        result = pd.concat([result, running(ridge, "Ridge")], ignore_index=True)

        return result
    #results = results.select(concat([results, variance_threshold(X)], ignore_index=True))
    results = results.select(concat([results, lasso(X, y)], ignore_index=True))
    results = results.select(concat([results, RF_Imp(X_train, y_train, X_test, y_test)], ignore_index=True))
    results = results.select(concat([results, PMI(X_train, y_train, X_test, y_test)], ignore_index=True))
    results = results.select(concat([results, RFE(X_train, y_train, X_test, y_test)], ignore_index=True))
    results = results.select(concat([results, SFS(X_train, y_train, X_test, y_test)], ignore_index=True))

    results['selected'] = results['importance'].apply(lambda x: 1 if x > 0 else 0)

    th_quantile=0.5
    th_rf = results[results['methods'] == 'Feature Importance']['importance'].quantile(th_quantile)
    print('\nThreshold for Random Forest Feature Importance is {:0.4f} at {:0.0f}th percentile'.format(th_rf, th_quantile*100))
    results['selected'] = results.apply(lambda x: 0 if ((x.methods == 'Feature Importance') & (x.importance < th_rf))
                                                    else x.selected, axis=1)
                
    results['Method'] = results.apply(lambda x: 'Variance Threshold' if x.methods == 'Variance Threshold'
                                                                    else str(x.methods) + " - " + str(x.model), axis=1)

    results['Count']=results.groupby('feature').transform(lambda x: x.sum())['selected']
    results = results.sort_values(by=['Count', 'feature'], ascending=True)
                                                                
    return results   

@transform_pandas(
    Output(rid="ri.vector.main.execute.d46ac7dc-7741-4012-b4be-f38d0c1d5c5c"),
    procedure_occurrence=Input(rid="ri.foundry.main.dataset.f8826e21-741d-49bb-a7eb-47ea98bb2b5f")
)
def proc(procedure_occurrence):
    df = procedure_occurrence
    print(df.columns)
    cols = ['person_id', 'procedure_occurrence_id', 'procedure_concept_id', 'procedure_date', 'procedure_concept_name']
    df = df[cols]
    return df

@transform_pandas(
    Output(rid="ri.vector.main.execute.f149e156-617f-4256-8e76-3fdbe1eb1b49"),
    procedures_to_macrovisits=Input(rid="ri.foundry.main.dataset.246dc59e-727a-44b4-99c2-0b6599831e4d")
)
def proc_macro(procedures_to_macrovisits):
    df = procedures_to_macrovisits['person_id', 'procedure_occurrence_id','procedure_date', 'macrovisit_start_date', 'macrovisit_end_date']
    print(df.columns)
    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ce49c9a4-0100-4401-8015-8c4b250c56b7"),
    deceasedSet=Input(rid="ri.foundry.main.dataset.74e51259-2e8a-4744-95b3-2a5b7c1bdb66")
)
def sample(deceasedSet):
    df = deceasedSet
    sam = df.sample(withReplacement=False, fraction=0.2)
    return sam

@transform_pandas(
    Output(rid="ri.vector.main.execute.f67c57de-ebb1-40e2-871a-aa033a37d8cb")
)
def unna(RandomForestCon5):
    df = RandomForestCon5
    df.groupBy('deceased', 'prediction').count()
    TN = df.where((df.prediction == '0') & (df.deceased == df.prediction)).count()
    TP = df.where((df.prediction == '1') & (df.deceased == df.prediction)).count()
    FN = df.where((df.prediction == '0') & (df.deceased != df.prediction)).count()
    FP = df.where((df.prediction == '1') & (df.deceased != df.prediction)).count()

    print('TN' , TN)
    print('TP' , TP)
    print('FN' , FN)
    print('FP' , FP)

@transform_pandas(
    Output(rid="ri.vector.main.execute.bebdaeff-5f8a-425f-8325-f4b4100b135c"),
    RandomForest4=Input(rid="ri.vector.main.execute.ca811206-918a-411c-920c-25e72ca445fa")
)
def unnam(RandomForestCon4):
    df = RandomForestCon4
    df.groupBy('deceased', 'prediction').count()
    TN = df.where((df.prediction == '0') & (df.deceased == df.prediction)).count()
    TP = df.where((df.prediction == '1') & (df.deceased == df.prediction)).count()
    FN = df.where((df.prediction == '0') & (df.deceased != df.prediction)).count()
    FP = df.where((df.prediction == '1') & (df.deceased != df.prediction)).count()

    print('TN' , TN)
    print('TP' , TP)
    print('FN' , FN)
    print('FP' , FP)

@transform_pandas(
    Output(rid="ri.vector.main.execute.4fd199c5-300f-4f6a-9e1a-a5f586f18b15"),
    RandomForest3=Input(rid="ri.foundry.main.dataset.7bf4046c-9e23-48eb-9a78-a3dbd43e866f")
)
def unnamed(RandomForestCon3):
    df = RandomForestCon3
    df.groupBy('deceased', 'prediction').count()
    TN = df.where((df.prediction == '0') & (df.deceased == df.prediction)).count()
    TP = df.where((df.prediction == '1') & (df.deceased == df.prediction)).count()
    FN = df.where((df.prediction == '0') & (df.deceased != df.prediction)).count()
    FP = df.where((df.prediction == '1') & (df.deceased != df.prediction)).count()

    print('TN' , TN)
    print('TP' , TP)
    print('FN' , FN)
    print('FP' , FP)

@transform_pandas(
    Output(rid="ri.vector.main.execute.79fcb3aa-2577-428a-a21a-436f0a7263f4"),
    sample=Input(rid="ri.foundry.main.dataset.ce49c9a4-0100-4401-8015-8c4b250c56b7")
)
def unnamed_1(sample):
    

