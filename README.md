# Bank Customer Churn Prediction

## Overview

This project aimed to develop a supervised machine learning model to predict customer churn at a fictitious bank in Europe. The dataset used for the model development was sourced from company's profile on [kaggle]("https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset"). After data inspection, EDA, and testing several models, I was able to build a model that can predict customer churn with 77% accuracy on the test set.

## Tools Used

* Python
* Pandas
* Numpy
* Scikit-Learn
* Matplotlib
* Seaborn

## About the dataset

* **Data Source**: The data used for the model development was sourced from kaggle via this [link]("https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset"). The data contains 10,000 records of the bank's customer data.

* **Data Inspection**: The data was properly inspected for any data quality issues such as, duplicates, missing values, errorneous values, data type mismatch, and inconsistency in categorical columns.

## EDA

The EDA revealed that 2,000 customers left the bank out of 10,000 customers, making the problem an imbalanced classification problem. This analysis also revealed the churn is common among older customers, customer that use more than two products, and German customers.

## Model Development

* **Data splitting**: The model was evaluted using stratified splitting and cross validation method so that the models performance can be properly measured for different subsets of the dataset.

* **Evaluation Metrics**: The model was evaluted using accuracy, precision, and recall scores. However the due the nature of the problem, recall was used as the primary metric for selecting the best model because this will help reduce the customer acquisition cost in cases of False Negatives.

* **Models**:
    Compared the performance of various models such as logistic regression, decison tree, random forest, gradient boosting, to a baseline model that predicts that no custmor will leave. The gradient boosting model which has an accuracy, recall, and precision scores of approximately 0.86, 0.48, and 0.75 respectively on the test set was chosen as the best model. Add cluster labels from KMeans algorithm slightly improved the models recall from approximately 0.48 to 0.49.

    Due the imbalance in the class labels, the probability threshold of the model was furthered tuned to maximaize the recall of the model on both the customers that left (True Positives) and customers that stayed (True Negatives). This reduced the models accuracy from approximately 0.86 to 0.76, and precision from 0.75 to 0.48, however this improved recall from 0.49 to 0.77.

## Business impact

In other to understand how the model helps the bank save money on customer aquisition cost (CAC). I compared the CAC and customer retention cost (CRC) of the baseline model which has an accuracy of 0.79, precision of 0.0, and recall of 0.0 on the test data and the best model with an accuracy of 0.86, precision of 0.46, and recall of 0.77 on the test data.

Let's assume the bank spends 1000 euros to acquire a customer and 200 euros to retain a customer. Using the confusion matrix of the test set, I compared the cost of the base model's error to the final models error. The confusion matrix shows the frequency of correct and incorrect predictions between the predicted outcomes and true outcomes.

Confusion Matrix of Baseline Model:

|                   | Predicted: Churn | Predicted: Won't Churn |
| :---------------- | :----------------- | :----------------- |
| Actual: Churn | 0 | 407 |
| Actual: Didn't Churn | 0 | 1593 |

Confusion Matrix of Best ML Model:

|                   | Predicted: Churn | Predicted: Won't Churn |
| :---------------- | :----------------- | :----------------- |
| Actual: Churn | 314 | 93 |
| Actual: Didn't Churn | 354 | 1236 |

From the confusion matrix of the two models, the bank would have hypothetically spend 407,000 euros on CAC if the base model was adopted, and 93,000 Euros on CAC if the best machine learning model was adopted. This shows that the machine learning model could reduce CAC by 77%.

## Next Steps

* Try other type of models, find the optimal parameters of the best model, or use oversampling and undersampling techniques to see if we can improve the recall and precision the model further.

* Collaborate with machine learning or software engineers so that the model can be easily accessible to the marketing department.