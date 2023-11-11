from pyspark.sql.functions import col,isnan, when, datediff, count, flatten, concat, collect_list
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, CountVectorizer, CountVectorizerModel
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
from pyspark.ml.feature import FeatureHasher

from sklearn.metrics import confusion_matrix
from pyspark.ml.clustering import LDA
#each item of a collection is modeled as a finite mixture over an underlying set of topics. each topic is modeled as infinite mixture over an underlying set of topic probablities
#decomposes the concept vector into two smaller parts: topic matrix and topic word

from foundry_ml import Model, Stage

@transform_pandas(
    Output(rid="ri.vector.main.execute.00b3ed2a-38ba-4046-b196-be710baeaaf9"),
    condition_era=Input(rid="ri.foundry.main.dataset.0d98e4e3-23a2-43b7-9fe7-db676ec45f84")
)
def ConditionsID(condition_era):
    df = condition_era
    print(df.columns)
    patientCount = df.select('person_id').distinct().count()
    conditionCount = df.select('condition_concept_id').distinct().count()
    print("# of patients:", patientCount)
    print("# of conditions:",conditionCount)
    result = df.groupBy('person_id').agg(collect_list('condition_concept_id').alias('conditions'))

    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e849e542-ca1f-4301-be31-19909a39ad75"),
    condition_era=Input(rid="ri.foundry.main.dataset.0d98e4e3-23a2-43b7-9fe7-db676ec45f84")
)
def ConditionsName(condition_era):
    df = condition_era
    print(df.columns)
    patientCount = df.select('person_id').distinct().count()
    conditionCount = df.select('condition_concept_name').distinct().count()
    print("# of patients:", patientCount)
    print("# of conditions:",conditionCount)
    result = df.groupBy('person_id').agg(collect_list('condition_concept_name').alias('conditions'))

    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.044fcc56-8d76-4632-b417-6001f2c048e6"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.526c0452-7c18-46b6-8a5d-59be0b79a10b")
)
def ConditionsName2(condition_occurrence):
    df = condition_occurrence
    print(df.columns)
    patientCount = df.select('person_id').distinct().count()
    conditionCount = df.select('condition_concept_name').distinct().count()
    print("# of patients:", patientCount)
    print("# of conditions:",conditionCount)
    result = df.groupBy('person_id').agg(collect_list('condition_concept_name').alias('conditions'))

    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.16cec201-32f7-48d8-81e9-938a78065368"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    drug_era=Input(rid="ri.foundry.main.dataset.4f424984-51a6-4b10-9b2b-0410afa1b2f8")
)
def DrugID1(drug_era, death):
    df = drug_era['person_id','drug_concept_id']
    
    patientCount = df.select('person_id').distinct().count()
    print(patientCount)
    count = df.select('drug_concept_id').distinct().count()
    print(count)
    result = df.groupBy('person_id').agg(collect_list('drug_concept_id').alias('drugs'))
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.47752beb-395b-41f1-89b0-9f5c8f619373"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.fd499c1d-4b37-4cda-b94f-b7bf70a014da")
)
def DrugID2(drug_exposure, death):
    df = drug_exposure['person_id','drug_concept_id']
    
    patientCount = df.select('person_id').distinct().count()
    print(patientCount)
    count = df.select('drug_concept_id').distinct().count()
    print(count)
    result = df.groupBy('person_id').agg(collect_list('drug_concept_id').alias('drugs'))
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.9e71356e-13b7-4fd5-840b-30c96dc17998"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    drug_era=Input(rid="ri.foundry.main.dataset.4f424984-51a6-4b10-9b2b-0410afa1b2f8")
)
def DrugName1(drug_era, death):
    df = drug_era['person_id','drug_concept_name']
    
    patientCount = df.select('person_id').distinct().count()
    print(patientCount)
    count = df.select('drug_concept_name').distinct().count()
    print(count)
    result = df.groupBy('person_id').agg(collect_list('drug_concept_name').alias('drugs'))
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.8e63496e-e7c4-417b-9901-b835662939dc"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    drug_exposure=Input(rid="ri.foundry.main.dataset.fd499c1d-4b37-4cda-b94f-b7bf70a014da")
)
def DrugsName2(drug_exposure, death):
    df = drug_exposure
    patientCount = df.select('person_id').distinct().count()
    print(patientCount)
    count = df.select('drug_concept_name').distinct().count()
    print(count)

    result = df.groupBy('person_id').agg(collect_list('drug_concept_name').alias('drugs'))
    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.2cde87af-5747-4f01-a131-3a20fae8066a"),
    vectorized=Input(rid="ri.foundry.main.dataset.bc7523c8-e461-4e37-a908-c525473116ef")
)

def LDAmodel(vectorized):
    df = vectorized
    lda = LDA(k = 20, featuresCol = 'concept_vector')
    model = lda.fit(df)
    #getting learned topics
    print(model.describeTopics())

    return Model(Stage(model))

@transform_pandas(
    Output(rid="ri.vector.main.execute.0f24115d-fc56-4028-b3bf-a1d2396365c7"),
    vectorized2=Input(rid="ri.foundry.main.dataset.b9aef32e-af33-48b4-a660-b5f867c8ca69")
)
from foundry_ml import Model, Stage
def LDAmodel2(vectorized2):
    df = vectorized2
    lda = LDA(k = 20, featuresCol = 'concept_vector')
    model = lda.fit(df)
    #getting learned topics
    print(model.describeTopics())

    return Model(Stage(model))

@transform_pandas(
    Output(rid="ri.vector.main.execute.e90f9ed6-221c-414b-bec2-2c988f53bc90"),
    vectorized3=Input(rid="ri.vector.main.execute.94ac7084-0a47-4e70-adc6-1a81b343fa85")
)
from foundry_ml import Model, Stage
def LDAmodel3(vectorized3):
    df = vectorized3
    lda = LDA(k = 20, featuresCol = 'concept_vector')
    model = lda.fit(df)
    #getting learned topics
    print(model.describeTopics())

    return Model(Stage(model))

@transform_pandas(
    Output(rid="ri.vector.main.execute.a9936b89-f80e-41cf-be23-a791344df345"),
    vectorized4=Input(rid="ri.foundry.main.dataset.c8d6ba08-dd9d-4a31-8d90-3e59f597dcf2")
)
from foundry_ml import Model, Stage
def LDAmodel4(vectorized4):
    df = vectorized4
    lda = LDA(k = 20, featuresCol = 'concept_vector')
    model = lda.fit(df)
    #getting learned topics
    print(model.describeTopics())

    return Model(Stage(model))

@transform_pandas(
    Output(rid="ri.vector.main.execute.bed1b4c4-553d-4b83-b086-8e45e7ffae59"),
    vectorized5=Input(rid="ri.vector.main.execute.8f746b74-fcbe-48fe-b001-751bd53a635a")
)
from foundry_ml import Model, Stage
def LDAmodel5(vectorized5):
    df = vectorized5
    lda = LDA(k = 20, featuresCol = 'concept_vector')
    model = lda.fit(df)
    #getting learned topics
    print(model.describeTopics())

    return Model(Stage(model))

@transform_pandas(
    Output(rid="ri.vector.main.execute.527881bc-41c6-47a4-be1e-29491b35639d"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    observation=Input(rid="ri.foundry.main.dataset.6ed38f3b-3fc9-4131-89d9-e443bb8c54fc")
)
def ObservationsID(observation, death):
    df = observation['person_id','observation_concept_id']

    patientCount = df.select('person_id').distinct().count()
    conditionCount = df.select('observation_concept_id').distinct().count()
    print("# of patients:", patientCount)
    print("# of observations:",conditionCount)
    result = df.groupBy('person_id').agg(collect_list('observation_concept_id').alias('observations'))
    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.cf8f7c33-1933-42e5-9e1a-5ab3c18a4e32"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    observation=Input(rid="ri.foundry.main.dataset.6ed38f3b-3fc9-4131-89d9-e443bb8c54fc")
)
def ObservationsName(observation, death):
    df = observation['person_id','observation_concept_name']

    patientCount = df.select('person_id').distinct().count()
    conditionCount = df.select('observation_concept_name').distinct().count()
    print("# of patients:", patientCount)
    print("# of observations:",conditionCount)
    result = df.groupBy('person_id').agg(collect_list('observation_concept_name').alias('observations'))
    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.9ccba7d5-4896-4f72-a241-b3a3f0338aa9"),
    agePredictions=Input(rid="ri.foundry.main.dataset.cffb04fd-c66e-44f9-b5d4-5da7dbfe3f63"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69"),
    person_=Input(rid="ri.foundry.main.dataset.af5e5e91-6eeb-4b14-86df-18d84a5aa010")
)
def Patient(person_, agePredictions, death):
    missing = agePredictions.select('person_id', 'year_of_birth', 'gender_concept_name', 'race_concept_name', 'ethnicity_concept_name','is_age_90_or_older', 'prediction') #missing subset
    
    df = person_.select('person_id', 'year_of_birth', 'gender_concept_name', 'race_concept_name', 'ethnicity_concept_name','is_age_90_or_older').dropna()
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
    mergedPatient = encoded.union(missing).drop_duplicates()

    df = mergedPatient
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

    df = df.withColumn('ageGroup_infant', when(col('age') < 2, 1)).withColumn('ageGroup_toddler', when(((col('age') >= 2) & (col('age') < 4)), 1)).withColumn('ageGroup_adolescent', when(((col('age') >= 4) & (col('age') < 14)), 1)).withColumn('ageGroup_youngAd', when(((col('age') >= 14) & (col('age') < 30)), 1)).withColumn('ageGroup_adult', when(((col('age') >= 30) & (col('age') < 50)), 1)).withColumn('ageGroup_olderAd', when(((col('age') >= 50) & (col('age') < 90)), 1)).withColumn('ageGroup_elderly', when((col('is_age_90_or_older') == 1), 1)).fillna(0)

    return df

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.f3880fa1-fb00-4837-abce-97860a3b25f2"),
    condition_occurrence=Input(rid="ri.foundry.main.dataset.526c0452-7c18-46b6-8a5d-59be0b79a10b"),
    death=Input(rid="ri.foundry.main.dataset.9c6c12b0-8e09-4691-91e4-e5ff3f837e69")
)
from pyspark.conf import SparkConf
def conditionsID2(condition_occurrence, death):

    df = condition_occurrence
    patientCount = df.select('person_id').distinct().count()
    conditionCount = df.select('condition_concept_id').distinct().count()
    print("# of patients:", patientCount)
    print("# of conditions:",conditionCount)

    result = df.groupBy('person_id').agg(collect_list('condition_concept_id').alias('condition'))
    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.71ffd1a9-f5f6-4357-881f-f0b5aa845f67"),
    ConditionsName=Input(rid="ri.foundry.main.dataset.e849e542-ca1f-4301-be31-19909a39ad75"),
    topics=Input(rid="ri.vector.main.execute.5d125945-c639-4e38-b3a8-616656ba8dd2")
)
from pyspark.sql.functions import udf
from pyspark.sql.types import *
def decoded(topics, ConditionsName):
    df = ConditionsName
    cv = CountVectorizer(inputCol = 'conditions', outputCol = 'concept_vector')
    model = cv.fit(ConditionsName)
    vocab = model.vocabulary

    def indices_to_vocab(indices_array):
        terms = []
        for i in indices_array:
            term = vocab[i]
            terms.append(term)
        
        return terms

    map_udf = udf(indices_to_vocab, ArrayType(StringType()))

    result = topics.withColumn("termTerms", map_udf("termIndices"))

    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e2d7c8d3-0d8e-4194-8bb5-6c8cf6c5ee83"),
    ObservationsName=Input(rid="ri.foundry.main.dataset.cf8f7c33-1933-42e5-9e1a-5ab3c18a4e32"),
    topics2=Input(rid="ri.vector.main.execute.3c88380d-b33e-4ae6-8520-3a95ab1a5192")
)
from pyspark.sql.functions import udf
from pyspark.sql.types import *
def decoded2(topics2, ObservationsName):
    df = ObservationsName
    cv = CountVectorizer(inputCol = 'observations', outputCol = 'concept_vector')
    model = cv.fit(ObservationsName)
    vocab = model.vocabulary

    def indices_to_vocab(indices_array):
        terms = []
        for i in indices_array:
            term = vocab[i]
            terms.append(term)
        
        return terms

    map_udf = udf(indices_to_vocab, ArrayType(StringType()))

    result = topics2.withColumn("termTerms", map_udf("termIndices"))

    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.78cc08a9-6a88-4b85-8cf5-a235e60591b6"),
    ConditionsName2=Input(rid="ri.vector.main.execute.044fcc56-8d76-4632-b417-6001f2c048e6"),
    topics3=Input(rid="ri.vector.main.execute.78307d0d-a125-41bf-a141-9fcbbd93880c")
)
from pyspark.sql.functions import udf
from pyspark.sql.types import *
def decoded3(topics3, ConditionsName2):
    df = ConditionsName2
    cv = CountVectorizer(inputCol = 'conditions', outputCol = 'concept_vector')
    model = cv.fit(ConditionsName2)
    vocab = model.vocabulary

    def indices_to_vocab(indices_array):
        terms = []
        for i in indices_array:
            term = vocab[i]
            terms.append(term)
        
        return terms

    map_udf = udf(indices_to_vocab, ArrayType(StringType()))

    result = topics3.withColumn("termTerms", map_udf("termIndices"))

    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.2dd9364a-22e3-4ef1-80eb-c6b42a9eb4b5"),
    DrugName1=Input(rid="ri.vector.main.execute.9e71356e-13b7-4fd5-840b-30c96dc17998"),
    topics4=Input(rid="ri.vector.main.execute.0e377b62-60b3-48f0-98d8-dc60349b75b9")
)
from pyspark.sql.functions import udf
from pyspark.sql.types import *
def decoded4(topics4, DrugName1):
    df = DrugName1
    cv = CountVectorizer(inputCol = 'drugs', outputCol = 'concept_vector')
    model = cv.fit(DrugName1)
    vocab = model.vocabulary

    def indices_to_vocab(indices_array):
        terms = []
        for i in indices_array:
            term = vocab[i]
            terms.append(term)
        
        return terms

    map_udf = udf(indices_to_vocab, ArrayType(StringType()))

    result = topics4.withColumn("termTerms", map_udf("termIndices"))

    return result

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d7d900fa-f814-436a-a739-89e300d34077"),
    DrugsName2=Input(rid="ri.vector.main.execute.8e63496e-e7c4-417b-9901-b835662939dc"),
    topics5=Input(rid="ri.vector.main.execute.d5aafd3c-d84b-442a-9544-5877c999f0ad")
)
from pyspark.sql.functions import udf
from pyspark.sql.types import *
def decoded5(topics5, DrugsName2):
    df = DrugsName2
    cv = CountVectorizer(inputCol = 'drugs', outputCol = 'concept_vector')
    model = cv.fit(DrugsName2)
    vocab = model.vocabulary

    def indices_to_vocab(indices_array):
        terms = []
        for i in indices_array:
            term = vocab[i]
            terms.append(term)
        
        return terms

    map_udf = udf(indices_to_vocab, ArrayType(StringType()))

    result = topics5.withColumn("termTerms", map_udf("termIndices"))

    return result

@transform_pandas(
    Output(rid="ri.vector.main.execute.5d125945-c639-4e38-b3a8-616656ba8dd2"),
    LDAmodel=Input(rid="ri.vector.main.execute.2cde87af-5747-4f01-a131-3a20fae8066a")
)
def topics(LDAmodel):
    model = LDAmodel
    ldamodel = model.stages[0].model
    #Model(Stage(ldamodel)) - a pipeline of models contained in stages
    #Having pipelines of models in stages supports developing complex pipelines and reusing them or using them in other apps in the enclave

    topics_description = ldamodel.describeTopics(maxTermsPerTopic = 100) 

    return(topics_description)

@transform_pandas(
    Output(rid="ri.vector.main.execute.3c88380d-b33e-4ae6-8520-3a95ab1a5192"),
    LDAmodel2=Input(rid="ri.vector.main.execute.0f24115d-fc56-4028-b3bf-a1d2396365c7")
)
def topics2(LDAmodel2):
    model = LDAmodel2
    ldamodel = model.stages[0].model
    #Model(Stage(ldamodel)) - a pipeline of models contained in stages
    #Having pipelines of models in stages supports developing complex pipelines and reusing them or using them in other apps in the enclave

    topics_description = ldamodel.describeTopics(maxTermsPerTopic = 100) 

    return(topics_description)

@transform_pandas(
    Output(rid="ri.vector.main.execute.78307d0d-a125-41bf-a141-9fcbbd93880c"),
    LDAmodel3=Input(rid="ri.vector.main.execute.e90f9ed6-221c-414b-bec2-2c988f53bc90")
)
def topics3(LDAmodel3):
    model = LDAmodel3
    ldamodel = model.stages[0].model
    #Model(Stage(ldamodel)) - a pipeline of models contained in stages
    #Having pipelines of models in stages supports developing complex pipelines and reusing them or using them in other apps in the enclave

    topics_description = ldamodel.describeTopics(maxTermsPerTopic = 100) 

    return(topics_description)

@transform_pandas(
    Output(rid="ri.vector.main.execute.0e377b62-60b3-48f0-98d8-dc60349b75b9"),
    LDAmodel4=Input(rid="ri.vector.main.execute.a9936b89-f80e-41cf-be23-a791344df345")
)
def topics4(LDAmodel4):
    model = LDAmodel4
    ldamodel = model.stages[0].model
    #Model(Stage(ldamodel)) - a pipeline of models contained in stages
    #Having pipelines of models in stages supports developing complex pipelines and reusing them or using them in other apps in the enclave

    topics_description = ldamodel.describeTopics(maxTermsPerTopic = 100) 

    return(topics_description)

@transform_pandas(
    Output(rid="ri.vector.main.execute.d5aafd3c-d84b-442a-9544-5877c999f0ad"),
    LDAmodel5=Input(rid="ri.vector.main.execute.bed1b4c4-553d-4b83-b086-8e45e7ffae59")
)
def topics5(LDAmodel5):
    model = LDAmodel5
    ldamodel = model.stages[0].model
    #Model(Stage(ldamodel)) - a pipeline of models contained in stages
    #Having pipelines of models in stages supports developing complex pipelines and reusing them or using them in other apps in the enclave

    topics_description = ldamodel.describeTopics(maxTermsPerTopic = 100) 

    return(topics_description)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.bc7523c8-e461-4e37-a908-c525473116ef"),
    ConditionsName=Input(rid="ri.foundry.main.dataset.e849e542-ca1f-4301-be31-19909a39ad75")
)
def vectorized(ConditionsName):    
    # read in the python list of the vocab
    df = ConditionsName
    cv = CountVectorizer(inputCol = 'conditions', outputCol = 'concept_vector')
    model = cv.fit(df)
    vocab = model.vocabulary
    print(vocab)
    # fortunately it's possible to build a vectorizer from a pre-computed vocab list such as we have
    # I'm also using binary = True to tell it that when counting occurrances we don't really need counts, just presence/absence (better for EHR data? you decide...)
    cvmodel = CountVectorizerModel.from_vocabulary(vocab, inputCol = "conditions", outputCol = "concept_vector", binary = True)

    # now we do the 'transform' (vectorization) and return it
    transformed_data = cvmodel.transform(Conditions1)

    return transformed_data

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b9aef32e-af33-48b4-a660-b5f867c8ca69"),
    ObservationsName=Input(rid="ri.foundry.main.dataset.cf8f7c33-1933-42e5-9e1a-5ab3c18a4e32")
)
def vectorized2(ObservationsName):
    cv = CountVectorizer(inputCol = 'observations', outputCol = 'concept_vector')
    model = cv.fit(ObservationsName)
    vocab = model.vocabulary
    print(vocab)

    cvmodel = CountVectorizerModel.from_vocabulary(vocab, inputCol = "observations", outputCol = "concept_vector", binary = True)

    # now we do the 'transform' (vectorization) and return it
    transformed_data = cvmodel.transform(ObservationsName)

    return transformed_data

@transform_pandas(
    Output(rid="ri.vector.main.execute.94ac7084-0a47-4e70-adc6-1a81b343fa85"),
    ConditionsName2=Input(rid="ri.vector.main.execute.044fcc56-8d76-4632-b417-6001f2c048e6")
)
def vectorized3(ConditionsName2):
    cv = CountVectorizer(inputCol = 'conditions', outputCol = 'concept_vector')
    model = cv.fit(ConditionsName2)
    vocab = model.vocabulary
    print(vocab)

    
    cvmodel = CountVectorizerModel.from_vocabulary(vocab, inputCol = "conditions", outputCol = "concept_vector", binary = True)

    # now we do the 'transform' (vectorization) and return it
    transformed_data = cvmodel.transform(ConditionsName2)

    return transformed_data

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c8d6ba08-dd9d-4a31-8d90-3e59f597dcf2"),
    DrugName1=Input(rid="ri.vector.main.execute.9e71356e-13b7-4fd5-840b-30c96dc17998")
)
def vectorized4(DrugName1):
    df = DrugName1
    cv = CountVectorizer(inputCol = 'drugs', outputCol = 'concept_vector')
    model = cv.fit(DrugName1)
    vocab = model.vocabulary
    print(vocab)

    cvmodel = CountVectorizerModel.from_vocabulary(vocab, inputCol = 'drugs', outputCol = 'concept_vector', binary=True)

    trans = cvmodel.transform(DrugName1)
    return trans

@transform_pandas(
    Output(rid="ri.vector.main.execute.8f746b74-fcbe-48fe-b001-751bd53a635a"),
    DrugsName2=Input(rid="ri.vector.main.execute.8e63496e-e7c4-417b-9901-b835662939dc")
)
def vectorized5(DrugsName2):
    df = DrugsName2
    cv = CountVectorizer(inputCol = 'drugs', outputCol = 'concept_vector')
    model = cv.fit(DrugsName2)
    vocab = model.vocabulary
    print(vocab)

    model = CountVectorizerModel.from_vocabulary(vocab, inputCol = 'drugs', outputCol = 'concept_vector', binary=True)

    trans = model.transform(DrugsName2)
    return trans

