# Data README

* 10k_synthea_covid_19_csv: Synthea Data
* test : Learning Loss test files
## Enclave
### Modeling Improvements (Updated  Data)

|     Dataset                |          Rows        |     Org   Column #    |     Columns   Used    |     #   Patient    |     #   Concepts    |     #   Represented    |     Deceased    |
|----------------------------|:--------------------:|:---------------------:|:---------------------:|:------------------:|:-------------------:|:----------------------:|:---------------:|
|     person                 |       18,844,666     |           27          |            7          |      18,844,666    |           3         |            20          |      539,898    |
|     death                  |        539,898       |           11          |            1          |       539,898      |           1         |          label         |      539,898    |
|     condition_era          |      369,865,270     |            8          |            4          |      16,018,541    |          172        |           172          |      475,033    |
|     condition_occurence    |     1,501,045,034    |           21          |            3          |      16,718,137    |          524        |           524          |      488,133    |
|     observations           |     1,135,698,679    |           25          |            5          |      17,588,418    |          155        |           155          |      484,253    |
|     drug_exposure          |     2,546,430,059    |           28          |           28          |      15,520,199    |          732        |           732          |      96,379     |
|     drug_era               |      672,699,925     |            9          |            9          |      14,316,720    |          226        |           226          |      94,880     |




### Long Covid Computational Challenge
[DataSet Breakdown](https://git.txstate.edu/DataLab/L3C-NIH/blob/main/data/DataSet%20Breakdown.xlsx)

1. a) Censored Training Datasets (after initial cleaning)

|Dataset Names 	           |Columns |Rows      |
|--------------------------|--------|----------|
|care_site		   |4 	    |8,367     |
|condition_era             |8	    |2,484,521 |
|condition_occurence	   |13	    |6,495,866 |
|condition_to_macrovisits  |8	    |6,276     |
|device_exposure           |12	    |422,167   |
|drug_era		   |9	    |2,090,455 |
|drug_exposure		   |13	    |13,611,559|
|location		   |4	    |25,142    |
|long COVID		   |5	    |57,672    |
|manifest_safe_harbor      |5	    |69        |
|measurement		   |24	    |32,569,723|
|measurement_to_macrovisits|8       |17,839,906|
|microvisits_to_macrovisits|23      |3,524,398 |
|note			   |15      |321,151   |
|note_nlp		   |16      |7,580,262 |
|observation		   |20      |6,869,266 |
|observation_period	   |7       |45,404    |
|payer_plan_period	   |8       |1,370,746 |
|person			   |14      |54,671    |
|procedure_occurence	   |10      |2,785,981 |
|procedure_to_macrovisits  |8       |991,579   |
|provider		   |7       |31,664    |
|visit_occurence           |14	    |3,509,934 |


2. a) Censored Testing Datasets (after initial cleaning)

|Dataset Names 	           |Columns |Rows      |
|--------------------------|--------|----------|
|care_site		   |4 	    |6         |
|condition_era             |8	    |13.920    |
|condition_occurence	   |13	    |35,618    |
|condition_to_macrovisits  |8	    |6,276     |
|device_exposure           |12	    |3,355     |
|drug_era		   |9	    |10,781    |
|drug_exposure		   |13	    |95,468    |
|location		   |4	    |255       |
|manifest_safe_harbor      |5	    |41        |
|measurement		   |24	    |192,170   |
|measurement_to_macrovisits|8       |120,436   |
|microvisits_to_macrovisits|23      |16,393    |
|observation		   |20      |53,547    |
|observation_period	   |7       |238       |
|payer_plan_period	   |8       |8,945     |
|person			   |14      |300       |
|procedure_occurence	   |10      |12,618    |
|procedure_to_macrovisits  |8       |4,340     |
|provider		   |7       |311       |
|visit_occurence           |14	    |16,378    |
