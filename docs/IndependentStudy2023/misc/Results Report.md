# Independent Study Results Report
## N3C Clinical Tabular Data in the Wild: Data Science Modeling Improvements

**Mirna Elizondo, Computer Science Texas State University**

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

|     data   frame    |    Random Forest   |           |          |                   |            | Logistic Regression |               |       | Gradient Boosting |           |        |       |
|:-------------------:|:------------------:|:---------:|:--------:|:-----------------:|:----------:|:-------------------:|:-------------:|:-----:|:-----------------:|:---------:|:------:|-------|
|                     | Accuracy           | Precision | Recall   | F1                | Accuracy   | Precision           | Recall        | F1    | Accuracy          | Precision | Recall | F1    |
| condition_era       |        0.500       |   0.500   |   0.500  |       0.500       |    0.500   |        0.500        |     0.500     | 0.500 |       0.506       |   0.506   |  0.506 | 0.506 |
| condition_occurence |        0.500       |   0.500   |   0.500  |       0.500       |    0.500   |        0.500        |     0.500     | 0.500 |       0.545       |   0.545   |  0.545 | 0.545 |
| observations        |        0.500       |   0.500   |   0.500  |       0.500       |    0.972   |        0.962        |     0.972     | 0.959 |       0.615       |   0.615   |  0.615 | 0.615 |
| drug_exposure       |         n/a        |    n/a    |    n/a   |        n/a        |    0.969   |        0.939        |     0.969     | 0.954 |        n/a        |    n/a    |   n/a  |  n/a  |
| drug_era            |         n/a        |    n/a    |    n/a   |        n/a        |    0.967   |        0.935        |     0.967     | 0.951 |       0.515       |   0.515   |  0.515 | 0.515 |
|     drug_era        |     672,699,925    |      9    |     9    |     14,316,720    |     226    |          226        |     94,880    |       |                   |           |        |       |

### Feature Topics
From the current LDA soft-clustering model 20 topics were selected

Example:
| Topic | Index | Weight | Term                   |
|-------|-------|--------|------------------------|
| 2     | 0     | 0.018  | Essential hypertension |
| 16    | 0     | 0.02   | Essential hypertension |
| 7     | 0     | 0.009  | Essential hypertension |
