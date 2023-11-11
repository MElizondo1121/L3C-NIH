# Independent Study Final Report
## N3C Clinical Tabular Data in the Wild: Data Science Modeling Improvements

**Mirna Elizondo, Computer Science Texas State University**

## Problem Statement
As a continuation to this previous project, I will be using the N3C Enclave to identify the attributes that lead a patient to developing Covid-19, the pre-condition to Long COVID. By utilizing our previous pipeline I will be improving the feature selection and scalability using PySpark.
From this project we had 5 findings : (1) Not every file or column in N3C is relevant; (2) We no longer have the ‘Long Covid’ label but do a death dataset that corresponds to every patient that has died; (3) One patient can have multiple conditions, observations, and drugs per visit (occurrence or era); (4) Encoding the fields as binary or numerical had no impact on modeling performance across the data frames ; (4) Early aggregation decisions that impacted the outcome; (5) How to predict missing age using a Logistic Regression model. By individually analyzing the relevant files I hope to be able to improve the efficiency of our pipeline. In order to improve the aggregation process, I will be including all conditions, observations and medications; previously we had included those concepts that had more than one million patients per concept and file.

#### Where does the data come from?
[NIH N3C: Enclave](https://ncats.nih.gov/n3c)

#### What are its characteristics?

|     Dataset                |          Rows        |     Org   Column #    |     Columns   Used    |     #   Patient    |     #   Concepts    |     #   Represented    |     Deceased    |
|----------------------------|:--------------------:|:---------------------:|:---------------------:|:------------------:|:-------------------:|:----------------------:|:---------------:|
|     person                 |       18,844,666     |           27          |            7          |      18,844,666    |           3         |            20          |      539,898    |
|     death                  |        539,898       |           11          |            1          |       539,898      |           1         |          label         |      539,898    |
|     condition_era          |      369,865,270     |            8          |            4          |      16,018,541    |          172        |           172          |      475,033    |
|     condition_occurence    |     1,501,045,034    |           21          |            3          |      16,718,137    |          524        |           524          |      488,133    |
|     observations           |     1,135,698,679    |           25          |            5          |      17,588,418    |          155        |           155          |      484,253    |
|     drug_exposure          |     2,546,430,059    |           28          |           28          |      15,520,199    |          732        |           732          |      96,379     |
|     drug_era               |      672,699,925     |            9          |            9          |      14,316,720    |          226        |           226          |      94,880     |

#### What do you hope to achieve?

A more robust and informative model. Individual conditions, procedures, and drugs should be incorporated into the model. To determine the most impactful factors, different methods must be used to classify the importance of the features in all data frames.

## Related Work
Over the last three years of the acute Coronavirus disease (COVID-19), millions of deaths have occurred. Significant advances have been made in identifying and treating this disease; however, our understanding of the disease has only begun. Scientists have found that in some cases, patients experience post-acute sequelae of SARS-CoV-2 infection (Long COVID, PASC, post-COVID-19 condition), which appears as lasting or new symptoms four weeks before or after being diagnosed with COVID-19. The current definition of Long COVID needs to be clarified due to the need for a greater understanding of the disease. Long COVID is currently reported to have heterogeneous signs and symptoms that can make identifying the most important one difficult; many of these symptoms can appear in other diseases and conditions. Understanding the different conditions or combinations that can lead to Long COVID will allow scientists to develop specific treatments. Patients with autoimmune disorders have shown significant differences in the symptoms they experience as well as by age. As part of DataLab12, we recently participated in the Long COVID Computational Challenge hosted by the National COVID Cohort Collaborative in the Enclave system (N3C), a National Institute of Health (NIH) National Center for Advancing Translational Sciences (NCATS). In the N3C system, we had access to patient information provided by 75 healthcare centers and 49/50 states in the United States. Data from this challenge represent 15 million patients, including 5.8 million cases of COVID positive and more than 17.5 rows of data.

In a similar study conducted in March 2022 by various institutions, the N3C system contained 2,909,292 patients and 5,645 patients diagnosed with COVID of Long, following the U09.9 code. This provided data set was found and further confirmed that a steeper risk gradient for Long COVID increases depends on the severity of COVID-19 infection. Multi-source clinical tabular data are increasingly challenging to tackle on large scales, and Long COVID data are constantly expanding. This is seen in a study conducted by the N3C Consortium in June 2022. At that time, the data consisted of 1,793,604 patients and 97,995 patients diagnosed with Long COVID, following the U09.9 code. Similarly to the March study, age and sex resulted in high feature importance scores. However, in this study, they developed three XGBoost machine learning models compared to adapting The Phenomizer, which is a web application that generates "a list of clinical characteristics that are most specific for individual diagnoses in a set of selected syndromes and can use this list to guide the further study".

## Data Management

### Data Gathering
*Composition*
1. What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)?

-  In the N3C system, we have access to patient information provided by 75 healthcare centers and 49/50 states in the United States. Data from this challenge represent 15 million patients, including 5.8 million cases of COVID positive and more than 17.5 rows of data.

2. How many instances are there in total (of each type, if appropriate)?

- Demographics records:
  - person: 18,693,017 patients
- Conditions records:
  - condition_era: 59,707 conditions
  - condition_occurrence: 58,669 conditions
- Observations records for 16,908,022 patients
- Drugs records:
  - drug_era: 28,405 drugs
  - drug_exposure: 32,645 drugs
- Death records for 552,594 patients.


3. Is there a label or target associated with each instance?
- Patients will be labeled as either alive or dead, from the death file I created a list of deceased patients and used it to create the 'deceased' column that will be 0 is the patient is alive and 1 if the patient is recorded as deceased.

4. Is any information missing from individual instances?

- Not all institutions and data centers recorded the same amount of information, some information is missing due to this and can be predicted or removed for the study. The majority of collected information that was missing will be excluded but 755,897 patients are missing 'year_of_birth', they were predicted using a Logistic Regression model from 'raceIndex', 'genderIndex', 'ethnicityIndex', 'is_age_90_or_older'. Once the 'global_person_id' is implemented on the Enclave I plan on integrated more features to correctly predict age.

5. Are relationships between individual instances made explicit.
- Patients may have multiple conditions, drugs, observations, and/or procedures recorded in a single visit, era, or occurrence. The patient:concept can also have multiple recorded time durations.

6. Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.
- This data as previously seen in the L3C Challenge is extremely noisy since it is gathered from multiple institutions the specific way that the information is gathered various. For example, all female 'gender_concept_name' may be recorded as "F", "Female", "Fem", "1".

7. Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

- The data cannot be downloaded out of the Enclave system, to access it institutional acceptance is required and a valid DUR must be submitted based on the requested data level of the de-identified data. The Enclave itself is updated in regular cycles it does not remain the same. The exact data input into the code workbook remains the same 'version' but if the pipeline is implemented in the code repository there is an option to update the data imported along with the overall site updates.

8. Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor patient confidentiality?

- The data set is level 2 de-identified patient records. Specific identifiers have been removed or shortened (zip code: 3 digits) while dates may be shifted by a number of days to protect patient identification.

9. Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?
- All patients can be identified by their unique person_id, which is unique between files. Currently the Enclave is set to complete patient de-duplication which will allow for patients to be identified using a 'global_person_id'.  

### Data Pre-processing
The files included in this study are *condition_occurence, condition_era, observations, drug_era*, and *drug_exposure*. They are sampled at more than 1 million patients per concepts, I used the columns found to be relevant in our previous study and aggregated the total concept duration for each file, similar to the previous study I processed the *person* file to be able to predict the missing age values.

**Labeling**
I created the labels in each file to represent if the patient was recorded to be alive or deceased. The 'deceased' column in the dataframe will be labeled as (deceased=0) if the patient is alive and (deceased=1) if the patient is dead.

## Exploratory Data Analysis
Ages | Gender | Race
--- | --- | ---
![ages](https://git.txstate.edu/DataLab/L3C-NIH/blob/main/docs/IndependentStudy2023/figures/Ages.png) | ![gender](https://git.txstate.edu/DataLab/L3C-NIH/blob/main/docs/IndependentStudy2023/figures/Gender.png)| ![race](https://git.txstate.edu/DataLab/L3C-NIH/blob/main/docs/IndependentStudy2023/figures/Gender.png)

**Age Groups:**
* Infant: 423,960
* Toddler: 527,762
* Adolescent: 1,487,853
* Young Adult: 3,388,801
* Adult: 5,147,122
* Older Adult: 7,119,600
* Elderly: 17,621,396

```
    ageGroup_infant < 2
    ageGroup_toddler ('age') >= 2 & ('age') < 4)
    ageGroup_adolescent ('age') >= 4) & (col('age') < 14)
    ageGroup_youngAd ('age') >= 14) & (col('age') < 30)
    ageGroup_adult ('age') >= 30) & (col('age') < 50)
    ageGroup_olderAd ('age') >= 50) & (col('age') < 90)
    ageGroup_elderly ('is_age_90_or_older') == 'true')
```

## Tools and Infrastructure
[Tools](https://covid.cd2h.org/tools)
- R & Python: data manipulation, visualization, and predictive model development
  - Code Workbook: improves the discoverability of workflow
- Contour: N3C feature that allows for a top-down data analysis tool used to manipulate and graph datasets.
- Apache Spark: used for filtering, joining, and aggregating datasets in the Enclave natively with PySpark

For this project the main implementation will be done in **PySpark** specifically spark.ml is the primary Machine Learning API for Spark. MLlib provides classification, regression, clustering and collaborative filtering. For this project I will be implementing (3) feature selectors, (1) soft clustering model, (1) regression model and (3) classification models.

**Feature Selectors**
*
*
**Models**



### Existing Tools and Infrastructure Used

Describe the tools that you used and the reasons for their choice. Justify them in terms of the problem itself and the methods you want to use.


Using the N3C Enclave the transformation will be handled in the Code Workbook.

```
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, CountVectorizer, CountVectorizerModel
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.clustering import LDA
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator, MulticlassMetrics
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
```

Foundry tools were imported in order to pass the model transformation rather than a dataframe, this was an initial obstacle we were trying to solve.

```
from foundry_ml import Model, Stage
```


#### Tools and Infrastructure Tried and Not Used

Describe any tools and infrastructure that you tried and ended up not using.
What was the problem?

The feature tools I tried this time around were Pivot and FeatureHasher, in order to handle the number of concepts included in every individual file I needed to find a way to configure the Foundry environment to allow a dataframe of that size. In the prior project we found that in the Foundry environment certain packages were not available. Both of these functions were not adaptable in the current environment to handle the number of concepts. In a smaller dataset both of these were implemented and functioned.

* Pivot: pivots a column by a specified aggregation into multiple columns, the number of concepts outnumbered the allocated max (10,000) option.
* FeatureHasher: projects a set of features into a feature vector, this is done by hashing to map the features to indices in the vector.


The classification model tried was Factorization Machine is a predictor model that combines advantages of SVM and applies factorized parameters instead of dense parametrization like in SVM. I thought this would be an efficient model since the Foundry environment works more efficiently with dense vectors rather than sparse, unfortunately it seemed to fail on every run. (Finding the answers to why they fail in the Enclave is difficult, if a run fails it may cause all other processes to fail (ending with no log codes)). Running then solo works best but PySpark has decreased the number of failed runs.


## Data Processing Tools
PySpark processing tools offer simple and efficient functions that allow for the conversion from Python to PySpark to run smoothly.
```
from pyspark.sql.functions import col, isnan, when, datediff, count, flatten, concat, sum, first,  countDistinct, max
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
```

The significant pipeline improvement is integrated in feature processing:
I implemented a Latent Dirichlet Allocation (LDA) soft-clustering algorithm. The LDA model is considered 'soft' because LDA produces a distribution of groupings over the items being clustered, each document will be assigned to distribution of topics, with each topic having a probability associated with it. Previously I manually grouped the words by keywords which LDA finds topics based on frequency counts to determine similarity, unfortunately the number of groupings needs to be specified in advance. Future experiments can be done to test the best number of groupings (for now I have set 20 groups but it is too little).

```
from pyspark.sql.functions import collect_list
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.clustering import LDA
```

### Machine Learning Tools
In our previous study, the Random Forest Model scored highest among the rest and the Logistic Regression model was used as a baseline model, the same will be done here but I will also be implementing the Gradient Boosting Model and a Latent Dirichlet Allocation (LDA) soft-clustering algorithm. The LDA model is considered 'soft' because LDA produces a distribution of groupings over the items being clustered, each document will be assigned to distribution of topics, with each topic having a probability associated with it. Previously I manually grouped the words by keywords which LDA finds topics based on frequency counts to determine similarity, unfortunately the number of groupings needs to be specified in advance. Future experiments can be done to test the best number of groupings (for now I have set 20 groups but it is too little).


### Visualization Tools
- The Enclave itself offers a visualization tool but the figures created require acceptance, for convenience the bare minimum of is extracted and graphed using Excel. But I was able to get matplotlib with the recent updates so i will be implementing what is available.


### Sampled Modeling results

For each file the concepts were sampled to include those that had more than one million patients per concepts.

* sklearn confusion matrix

| Confusion Matrix|
|:-------:|:-----:|
| TN      | FP    |
| FN      | TP    |


**note** (those labeled n/a have been crashing but are fully functioning)

#### Confusion Matrix Results
|                     |         | condition_era | condition_occurence | observations | drug_exposure | drug_era |
|:-------------------:|:-------:|---------------|---------------------|--------------|---------------|----------|
|        Labels       | Class 0 |    15537259   |       16233297      |   17107179   |    3008694    | 13847578 |
|                     | Class 1 |     481282    |        484840       |    481239    |     96379     |  469142  |
|    Random Forest    |    TN   |    15537259   |         n/a         |   17107179   |      n/a      |    n/a   |
|                     |    FN   |     481282    |         n/a         |    481239    |      n/a      |    n/a   |
|                     |    FP   |       0       |         n/a         |       0      |      n/a      |    n/a   |
|                     |    TP   |       0       |         n/a         |       0      |      n/a      |    n/a   |
| Logistic Regression |    TN   |    15537259   |       16233297      |   17103293   |      n/a      |    n/a   |
|                     |    FN   |     481282    |        484840       |    483082    |      n/a      |    n/a   |
|                     |    FP   |       0       |          0          |      872     |      n/a      |    n/a   |
|                     |    TP   |       0       |          0          |     1171     |      n/a      |    n/a   |
|  Gradient Boosting  |    TN   |    3104870    |       3239600       |    3405768   |      n/a      |  2766665 |
|                     |    FN   |     94722     |        87848        |     73508    |      n/a      |   90666  |
|                     |    FP   |      886      |         5888        |     14313    |      n/a      |   1641   |
|                     |    TP   |      1224     |         8958        |     22456    |      n/a      |   2892   |

#### Measures per Model

|     data   frame    | Random Forest                                     | Logistic Regression |               | Gradient Boosting                           |
|:-------------------:|:-------------:|:---------:|:------:|:-----:|:--------:|:-------------------:|:------:|:-----:|:-----------------:|:---------:|:------:|-------|
|                     | Accuracy      | Precision | Recall | F1    | Accuracy | Precision           | Recall | F1    | Accuracy          | Precision | Recall | F1    |
| condition_era       |     0.500     |   0.500   |  0.500 | 0.500 |   0.500  |        0.500        |  0.500 | 0.500 |       0.506       |   0.506   |  0.506 | 0.506 |
| condition_occurence |     0.500     |   0.500   |  0.500 | 0.500 |   0.500  |        0.500        |  0.500 | 0.500 |       0.545       |   0.545   |  0.545 | 0.545 |
| observations        |     0.500     |   0.500   |  0.500 | 0.500 |   0.972  |        0.962        |  0.972 | 0.959 |       0.615       |   0.615   |  0.615 | 0.615 |
| drug_exposure       |      n/a      |    n/a    |   n/a  |  n/a  |   0.969  |        0.939        |  0.969 | 0.954 |        n/a        |    n/a    |   n/a  |  n/a  |
| drug_era            |      n/a      |    n/a    |   n/a  |  n/a  |   0.967  |        0.935        |  0.967 | 0.951 |       0.515       |   0.515   |  0.515 | 0.515 |


### Feature Topics
From the current LDA soft-clustering model 20 topics were selected

Example:
| Topic | Index | Weight | Term                   |
|-------|-------|--------|------------------------|
| 2     | 0     | 0.018  | Essential hypertension |
| 16    | 0     | 0.02   | Essential hypertension |
| 7     | 0     | 0.009  | Essential hypertension |

## Conclusion
