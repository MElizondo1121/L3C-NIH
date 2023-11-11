#ChatGPT Summary by sections
### Introduction

The COVID-19 pandemic has resulted in millions of deaths worldwide, and although progress has been made in identifying and treating the disease, there is still much to learn about COVID-19. Long COVID, a condition that manifests as lasting or new symptoms occurring four weeks before or after being diagnosed with COVID-19, has been identified as a significant concern. However, the current definition of Long COVID remains vague and requires further clarification. To gain a better understanding of the factors contributing to Long COVID, this project aims to utilize the N3C Enclave to enhance the feature selection and scalability of a previous pipeline through the use of PySpark.

### Problem Statement

This project builds upon a previous study that identified several findings regarding the N3C Enclave dataset, including the fact that not all files or columns are pertinent, patients can have multiple conditions, observations, and drugs per visit, and early aggregation decisions can impact the outcome. The project aims to improve the pipeline's efficiency by including all conditions, observations, and medications rather than just those with over one million patients per concept and file. By enhancing the feature selection and scalability of the previous pipeline through the use of PySpark, the project aims to analyze relevant files and determine the most impactful factors contributing to a patient's risk of developing COVID-19, which is a precursor to Long COVID.

### Objectives

The objectives of this project are to:

    Incorporate individual conditions, procedures, and drugs into the model to develop a more robust and informative model.
    Determine the most impactful factors contributing to a patient's risk of developing COVID-19 using different methods to classify the importance of the features in all data frames.

### Related Work

This study builds upon previous research that has been conducted on the N3C Enclave dataset. In March 2022, various institutions conducted a study that confirmed the risk of Long COVID increases with the severity of COVID-19 infection. The dataset for this study consisted of 2,909,292 patients and 5,645 patients diagnosed with Long COVID following the U09.9 code. In June 2022, the N3C Consortium conducted another study that found age and sex to have high feature importance scores. The researchers developed three XGBoost machine learning models and compared them to the Phenomizer, a web application that generates a list of clinical characteristics specific to individual diagnoses in selected syndromes to guide further study.

### Data Management

The N3C Enclave dataset contains patient information from 75 healthcare centers and 49/50 states in the United States, representing 15 million patients, including 5.8 million cases of COVID-19 positive patients, and more than 17.5 rows of data. The dataset is level 2 de-identified patient records that require institutional acceptance and a valid DUR to access. Dates may be shifted, and specific identifiers may be removed or shortened to protect patient identification. Patient de-duplication is to be completed in the Enclave to identify patients using a 'global_person_id.' The dataset is very noisy, and the information recorded can vary significantly, which requires careful consideration during the data analysis process


### Tools and Infrastructure Used

The project utilized several tools and infrastructure to facilitate data manipulation, visualization, and predictive model development. The following tools were used:

- R and Python: These were used for data manipulation, visualization, and predictive model development.
- Code Workbook: This tool improved the discoverability of workflow.
- Contour: This N3C feature allowed for top-down data analysis and manipulation of datasets.
- Apache Spark: This tool was used for filtering, joining, and aggregating datasets in the Enclave natively with PySpark.

For this project, the main implementation was done in PySpark, specifically using the spark.ml primary Machine Learning API for Spark. MLlib provided classification, regression, clustering, and collaborative filtering. The project implemented three feature selectors and utilized four modeling techniques: Random Forest, Logistic Regression (with regularization), Gradient Boosting, and Latent Dirichlet Allocation (LDA).

### Tried and Not Used

Two tools were tried but not used in the project. The first tool was the Pivot and FeatureHasher feature tools, which were tested to handle the large number of concepts in each individual file. However, these tools encountered issues with the size of the dataframe in the Foundry environment. In a smaller dataset, both functions worked well, but they were not adaptable to handle the large number of concepts in this project. Specifically, the Pivot function was unable to handle the number of concepts, as it exceeded the allocated maximum of 10,000 options. On the other hand, FeatureHasher function projects a set of features into a feature vector by mapping the features to indices in the vector using hashing. Nonetheless, it also encountered issues due to the large size of the dataset.

The second tool that was tried but not used in the project was a classification model called Factorization Machine. This model combines the strengths of Support Vector Machines (SVM) and factorized parameters instead of dense parameters like in SVM. The model was chosen because the Foundry environment works more efficiently with dense vectors than sparse ones. However, the model consistently failed in the Foundry environment, and it was challenging to determine why. When a run failed, it often caused other processes to fail without any log codes. Running the model solo was more successful, and using PySpark reduced the number of failed runs.

In conclusion, the tools and infrastructure used in the project allowed for efficient data manipulation, visualization, and predictive model development. Although two tools were tried and not used, the project team successfully adapted and utilized other tools and techniques to achieve the project goals.

Abstract

This report presents an analysis of the PySpark pipeline used in the L3C challenge. Specifically, we focus on the implementation of the Latent Dirichlet Allocation (LDA) soft-clustering algorithm in the feature processing stage and compare the performance of several machine learning models, including Logistic Regression, Random Forest, Gradient Boosting, and LDA. The results of our analysis show that the LDA soft-clustering algorithm offers a more automated and probabilistic way of grouping words into topics, providing interpretable results and uncovering latent structures that are not easily identifiable. Additionally, the Gradient Boosting model outperformed the other models for both condition and observation files, with the LDA soft-clustering model used to resolve data processing issues encountered in the L3C challenge.

Introduction

The L3C challenge is a collaborative effort between the National Institutes of Health (NIH) and the Observational Health Data Sciences and Informatics (OHDSI) community to develop methods for using electronic health record data to study patient outcomes. One of the key challenges in this effort is handling large-scale datasets, which require specialized tools and techniques for processing and analysis. In this report, we present an analysis of the PySpark pipeline used in the L3C challenge and explore the performance of several machine learning models in predicting patient outcomes.

Methodology

The feature processing stage of the PySpark pipeline was improved by implementing the LDA soft-clustering algorithm. We also compared the performance of several machine learning models, including Logistic Regression, Random Forest, Gradient Boosting, and LDA. The performance of these models was evaluated based on their ability to predict patient outcomes using condition and observation files.

Results

Our analysis showed that the implementation of the LDA soft-clustering algorithm in the feature processing stage provided a more automated and probabilistic way of grouping words into topics, producing interpretable results and uncovering latent structures that are not easily identifiable. The performance of the machine learning models varied, with the Gradient Boosting model outperforming the other models for both condition and observation files. The LDA soft-clustering model was used to resolve data processing issues encountered in the L3C challenge.

Conclusion

In conclusion, the implementation of the PySpark pipeline showed an improvement in runtime and reduced the number of failed runs, although there is still room for learning and improvement in data processing within the Enclave environment. The LDA soft-clustering algorithm provides a more automated and probabilistic way of grouping words into topics, providing interpretable results and uncovering latent structures that are not easily identifiable. The Gradient Boosting model outperformed the other models for both condition and observation files, with the LDA soft-clustering model used to resolve data processing issues encountered in the L3C challenge. Future work will focus on determining the correct ratio for topic:concept, investigating the removal of outliers that do not follow the trend in the figure countsPerConcept, and studying the effects of removing these features and implementing topic labels to improve runtime and modeling scores. Overall, the Enclave offers a convenient environment for handling large data and has potential for use in specific cohort studies.
