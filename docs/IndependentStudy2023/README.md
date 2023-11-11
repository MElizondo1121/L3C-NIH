## N3C Clinical Tabular Data in the Wild: Data Science Modeling Improvements

Authors: Mirna Elizondo and Jelena Tešić, Computer Science Department, Texas State University


Acknowledgements: The analyses described in this [publication/report/presentation] were conducted with data or tools accessed through the NCATS N3C Data Enclave https://covid.cd2h.org and N3C Attribution & Publication Policy v 1.2-2020-08-25b supported by NCATS U24 TR002306, Axle Informatics Subcontract: NCATS-P00438-B. This research was possible because of the patients whose information is included within the data and the organizations (https://ncats.nih.gov/n3c/resources/data-contribution/data-transfer-agreement-signatories) and scientists who have contributed to the on-going development of this community resource [https://doi.org/10.1093/jamia/ocaa196].

## Abstract:
The COVID-19 pandemic has created an urgent need to understand Long COVID, a condition that manifests as lasting or new symptoms occurring four weeks before or after being diagnosed with COVID-19. This project aims to enhance the feature selection and scalability of a previous pipeline through the use of PySpark to analyze relevant files and determine the most impactful factors contributing to a patient's risk of developing COVID-19, which is a precursor to Long COVID. The project utilizes the N3C Enclave dataset and incorporates individual conditions, procedures, and drugs into the model to develop a more robust and informative model. Three feature selectors and four modeling techniques, including Random Forest, Logistic Regression, Gradient Boosting, and Latent Dirichlet Allocation (LDA), were implemented in PySpark. The project team also utilized several tools and infrastructure, including R and Python for data manipulation, visualization, and predictive model development, and Apache Spark for filtering, joining, and aggregating datasets. Two tools, Pivot and FeatureHasher, were tried but not used due to issues with the size of the dataframe in the Foundry environment. Factorization Machine, a classification model, was also tried but consistently failed in the Foundry environment. The project team successfully adapted and utilized other tools and techniques to achieve the project goals. The results of this project could help clinicians better understand Long COVID's risk factors and improve the management and treatment of this condition.


## Problem Statement

The goal of this project is to build upon a previous project and utilize the N3C Enclave to identify the factors that contribute to a patient's risk of developing Covid-19, which is a precursor to Long COVID. The challenge lies in improving the feature selection and scalability of the previous pipeline to enhance the efficiency of data analysis.

To address the challenge, the project aims to analyze relevant files and improve the pipeline's efficiency by leveraging PySpark. The previous project revealed several key findings, such as the irrelevance of certain files or columns in the N3C dataset, the absence of a 'Long Covid' label, the possibility of multiple conditions, observations, and drugs per patient visit, and the inability of encoding fields as binary or numerical to impact modeling performance. The project also identified the impact of early aggregation decisions and the ability to predict missing ages using a Logistic Regression model. To enhance the aggregation process, the project will include all conditions, observations, and medications, rather than limiting to only those with over one million patients per concept and file.

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

During the past three years of the COVID-19 pandemic, millions of deaths have occurred globally. Although considerable progress has been made in identifying and treating the disease, there is still much to learn about COVID-19. Scientists have identified that some patients experience post-acute sequelae of SARS-CoV-2 infection, also known as Long COVID, which manifests as lasting or new symptoms occurring four weeks before or after being diagnosed with COVID-19. The current definition of Long COVID is still vague and needs further clarification.

Long COVID presents with various and diverse signs and symptoms that can make it difficult to pinpoint the most important one. Thus, understanding the different conditions or combinations that can lead to Long COVID is crucial for developing specific treatments. In particular, patients with autoimmune disorders may experience significant differences in their symptoms based on age and other factors.

As part of the DataLab12 project, we participated in the Long COVID Computational Challenge hosted by the National COVID Cohort Collaborative in the Enclave system (N3C). The N3C system provides access to patient information from 75 healthcare centers and 49/50 states in the United States, representing 15 million patients, including 5.8 million cases of COVID-19 positive patients and more than 17.5 rows of data.

In a similar study conducted by various institutions in March 2022, the N3C system contained 2,909,292 patients and 5,645 patients diagnosed with Long COVID following the U09.9 code. This dataset confirmed that the risk of Long COVID increases with the severity of COVID-19 infection. As multi-source clinical tabular data becomes increasingly challenging to manage on a large scale, and Long COVID data continually expands, the N3C Consortium conducted another study in June 2022. The dataset for this study consisted of 1,793,604 patients and 97,995 patients diagnosed with Long COVID following the U09.9 code. In this study, age and sex were found to have high feature importance scores. The researchers developed three XGBoost machine learning models and compared them to the Phenomizer, a web application that generates a list of clinical characteristics specific to individual diagnoses in selected syndromes to guide further study.


## Data Management

### Data Processing
The study includes several files, namely *condition_occurrence*, *condition_era*, *observations*, *drug_era*, and *drug_exposure*, with a sample size of over 1 million patients per concept. Relevant columns from the previous study were used, and the total concept duration for each file was aggregated. Missing age values were predicted using the *person* file, similar to the previous study. (1) The person dataset was used to create the demographics dataframe

The patients were labeled based on whether they were recorded as alive or deceased, with the 'deceased' column in the dataframe being assigned a value of 0 if the patient is alive and 1 if the patient is deceased. The following steps were taken: (1) all gender_concepts were mapped to 3 categories (Female, Male, Unknown); (2) all race_concepts&ethnicity_concepts were mapped into 10 different race values; and (3) we introduce age as a new predicate. For the age predictions we use a logistic regression model on *'race_concept_id', 'gender_concept_id', 'ethnicity_concept_id', 'is_age_90_or_older'*, which age as the reported difference between *death_date and year_of_birth*. This produces 24 feature columns in demographics.

The processing for the conditions, observations and medications dataframes were the same. For this project we only collected relevant columns and the focus was on the sampled set, the number of columns in each file is representative of the number of features included. Characteristics are seen in the table above.

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

For this project the main implementation will be done in **PySpark** specifically spark.ml is the primary Machine Learning API for Spark. MLlib provides classification, regression, clustering and collaborative filtering. For this project I be implementing (3) feature selectors and for modeling, we use 1. Random Forest 2. Logistic regression //with what regularization) 3. Gradient Boosting; and 4. Latent Dirichlet Allocation (LDA) 

```
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import UnivariateFeatureSelector
```


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

In this project, I tried to use Pivot and FeatureHasher feature tools to handle the large number of concepts in each individual file. However, I encountered issues with the size of the dataframe in the Foundry environment. In a smaller dataset, both functions worked well, but they were not adaptable to handle the large number of concepts in this project. Specifically, the Pivot function was unable to handle the number of concepts, as it exceeded the allocated maximum of 10,000 options. On the other hand, FeatureHasher function projects a set of features into a feature vector by mapping the features to indices in the vector using hashing. Nonetheless, it also encountered issues due to the large size of the dataset.


In this project, I experimented with a classification model called Factorization Machine, which combines the strengths of Support Vector Machines (SVM) and factorized parameters instead of dense parameters like in SVM. I chose this model because the Foundry environment works more efficiently with dense vectors than sparse ones. However, the model consistently failed in the Foundry environment, and it was challenging to determine why. When a run failed, it often caused other processes to fail without any log codes. Running the model solo was more successful, and using PySpark reduced the number of failed runs.


## Data Processing Tools
PySpark processing tools offer simple and efficient functions that allow for the conversion from Python to PySpark to run smoothly.
```
from pyspark.sql.functions import col, isnan, when, datediff, count, flatten, concat, sum, first,  countDistinct, max
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.sql.functions import collect_list
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from pyspark.ml.clustering import LDA
```


### Machine Learning Tools
In our previous study, we used several machine learning models such as Logistic Regression, Decision Trees, Random Forest, and Gradient Boosting. However, in this study, we have implemented an age prediction Logistic Regression (with regularization), a Random Forest and Gradient Boosting Classification Model, and Latent Dirichlet Allocation (LDA) Topic Model.

LDA is considered a 'soft' clustering algorithm because it assigns each document to a distribution of topics, where each topic has a probability associated with it. Unlike previous approaches where words were grouped manually by keywords, LDA uses frequency counts to determine similarity and groupings. However, one drawback is that the number of groupings needs to be specified in advance. In this implementation, we have set 20 groups, but it may not be the optimal number. Further experimentation is needed to determine the optimal number of groupings.

A notable improvement in the pipeline is the implementation of the LDA soft-clustering algorithm in the feature processing stage, which offers a more automated and probabilistic way of grouping words into topics.

For feature selection, we used Random Forest Feature Importance, Logistic Regression Feature Importance, and Univariate (Chi2) Feature Selector. As for the models, we used Logistic Regression, Random Forest, Gradient Boosting, and LDA.

* Age Prediction Model
  - Logistic regression is a simple and interpretable model that can handle both binary and multiclass classification problems. However, it assumes a linear relationship between features and the log-odds of the outcome, which may not be suitable for non-linear data or if there are irrelevant features in the dataset.

* Classification Modeling
  - Random Forest is a machine learning algorithm that uses multiple decision trees to make predictions. One of its advantages is its ability to achieve high accuracy in predicting outcomes. However, it may not be suitable for high-dimensional data, as the model's performance can decrease with an increasing number of features.
  - Gradient Boosting is an iterative algorithm that gradually learns to fit the data better over time. It is highly flexible and can handle a wide range of data types and formats, but can be computationally expensive and difficult to tune for very large datasets.

* Topic Modeling
  - LDA is a machine learning algorithm used for topic modeling, which aims to uncover latent topics within a collection of documents. Its advantages include the ability to handle large-scale datasets, producing interpretable results, and uncovering latent structures that are not easily identifiable. However, it can be sensitive to the choice of hyperparameters and may require a large number of topics to accurately capture the underlying patterns in the data.


### Visualization Tools
Although the Enclave provides a visualization tool, the figures generated require approval. Therefore, to streamline the process, only the essential information is extracted and graphed using Excel. However, with the recent updates, I was able to obtain matplotlib, and I plan to incorporate it into the visualization process.


### Sampled Modeling results

For each file the concepts were sampled to include those that had more than one million patients per concepts.

* sklearn confusion matrix

| Confusion Matrix| |
|:-------:|:-----:|
| TN      | FP    |
| FN      | TP    |


*note (those labeled n/a have been crashing but are fully functioning, be updating as they finish)*

#### Confusion Matrix Results
|                     |         | condition_era | condition_occurence | observations | drug_exposure | drug_era |
|:-------------------:|:-------:|:-------------:|:-------------------:|:------------:|:-------------:|:--------:|
|                     |  Train  |    12816839   |       13375843      |   14072373   |               |          |
|                     |   Test  |    3201702    |       3342294       |    3516045   |               |          |
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

|     data   frame    | Random Forest |           |        |       |          | Logistic Regression |        |       | Gradient Boosting |           |        |       |
|:-------------------:|:-------------:|:---------:|:------:|:-----:|:--------:|:-------------------:|:------:|:-----:|:-----------------:|:---------:|:------:|-------|
|                     | Accuracy      | Precision | Recall | F1    | Accuracy | Precision           | Recall | F1    | Accuracy          | Precision | Recall | F1    |
| condition_era       |     0.500     |   0.500   |  0.500 | 0.500 |   0.500  |        0.500        |  0.500 | 0.500 |       0.506       |   0.506   |  0.506 | 0.506 |
| condition_occurence |     0.500     |   0.500   |  0.500 | 0.500 |   0.500  |        0.500        |  0.500 | 0.500 |       0.545       |   0.545   |  0.545 | 0.545 |
| observations        |     0.500     |   0.500   |  0.500 | 0.500 |   0.972  |        0.962        |  0.972 | 0.959 |       0.615       |   0.615   |  0.615 | 0.615 |
| drug_exposure       |      n/a      |    n/a    |   n/a  |  n/a  |   0.969  |        0.939        |  0.969 | 0.954 |        n/a        |    n/a    |   n/a  |  n/a  |
| drug_era            |      n/a      |    n/a    |   n/a  |  n/a  |   0.967  |        0.935        |  0.967 | 0.951 |       0.515       |   0.515   |  0.515 | 0.515 |


### Feature Topics (Main Improvement)
Currently, the LDA soft-clustering model has identified 20 topics (although further processing and cleaning are needed to assign appropriate labels to them). In this soft-clustering model using Latent Dirichlet Allocation (LDA), topics are essentially clusters of words that frequently appear together across multiple documents. Each document can be considered a blend of these topics, with each topic contributing to the overall content of the document to some extent. The degree to which a topic influences a document is expressed through a probability distribution, where higher probabilities indicate a stronger correlation with that topic. The topics themselves do not have a specific interpretation or meaning, as they are generated solely from the data and are typically labeled based on the most common words that appear within them.

The main objective of the LDA model is to automate the process of grouping related medical conditions without selecting any specific word or code. This model can be particularly useful for preprocessing features into groups and subsequently utilizing feature selectors to identify a subset of concepts that may be associated with a patient's likelihood of mortality.

[Topics Per File w/ first term](https://git.txstate.edu/DataLab/L3C-NIH/blob/main/docs/IndependentStudy2023/misc/TopicsResultsReport.md)

Example:
| Topic | Indices* | Weights* | Terms*            |
|-------|-------|--------|------------------------|
| 2     | 0     | 0.018  | Essential hypertension |
| 16    | 0     | 0.02   | Essential hypertension |
| 7     | 0     | 0.009  | Essential hypertension |


**Note: The identifiers array is shown with one value for simplicity.**

## Conclusion
The implementation of the PySpark pipeline showed an improvement in runtime and reduced the number of failed runs, although there is still room for learning and improvement in data processing within the Enclave environment. The Enclave offers a convenient environment for handling large data and has potential for use in specific cohort studies. The modeling results provided insight into which models fit the data best, with the Gradient Boosting model outperforming the Random Forest and Logistic Regression models for both condition and observation files, but the Observation models are producing nonrandom results. However, due to the system updates and data request collisions, the drug_exposure Gradient Boosting model caused the other models to crash. The LDA soft-clustering model was used to resolve data processing issues encountered in the L3C challenge. In future work, the correct ratio for topic:concept will be determined, and investigating the removal of outliers that do not follow the trend in the figure [countsPerConcept](https://git.txstate.edu/DataLab/L3C-NIH/blob/main/docs/IndependentStudy2023/figures/CountsPerConcept.png) could lead to modeling improvements. Additionally, studying the effects of removing these features and implementing topic labels could improve runtime and modeling scores.
