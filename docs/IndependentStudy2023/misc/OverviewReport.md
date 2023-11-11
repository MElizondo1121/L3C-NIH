# Independent Study Overview Report
## N3C Clinical Tabular Data in the Wild: Data Science Modeling Improvements

**Mirna Elizondo, Computer Science Texas State University**

From this project we learned kept five facts in mind: (1) Not every file or column in N3C is relevant; (2) We no longer have the ‘Long Covid’ label but do a death dataset that corresponds to every patient that has died; (3) One patient can have multiple conditions, observations, and drugs per visit (occurrence or era); (4) Encoding the fields as binary or numerical had no impact on modeling performance across the data frames ; (4) Early aggregation decisions that impacted the outcome; (5) How to predict missing age using a Logistic Regression model. By individually analyzing the relevant files we are able to improve the efficiency of our pipeline. In order to improve the aggregation process, I will be including all conditions, observations and medications; previously we had included those concepts that had more than one million patients per concept and file.

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

Include informal success measures (e.g. accuracy on cross-validated data, without specifying ROC or precision/recall etc.) that you planned to use.

- Accuracy, Precision, Recall, F1 scores
- Calculating TP, TN, FP, FN using a confusion matrix

#### What do you hope to achieve?

A more robust and informative model. Individual conditions, procedures, and drugs should be incorporated into the model. To determine the most impactful factors, different methods must be used to classify the importance of the features in all data frames.

#### What improvements were implemented into the pipeline?

I have created a LDA Clustering Model that does not assume any distance measure between topics but instead it infers topics based on words counts (bag-of-words). The word probabilities are maximized by dividing the words among the topics but currently with the topic # (20) is seen to be to low as was expected. Experiments can be done to find the appropriate number of topics as it relates to patient counts and specific class distribution in order to assist the class imbalances. The key is finding the appropriate topic proportions. In the case of simply analyzing features I found that the aggregation of concepts can be better analyzed with a clustering method.
