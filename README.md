# Bank Marketing Campaign: Comparing Classification Models



## Project Overview



This project uses supervised machine learning to predict whether a bank
customer will subscribe to a term deposit after being contacted as part
of a direct marketing campaign.

The analysis was completed in **Python using Google Colab**.



## Business Objective

The business objective is to help a bank make its direct marketing
campaigns more efficient by identifying customers who are more likely to
subscribe to a term deposit.

A useful model could help the bank prioritize its outreach rather than
treating every customer as equally likely to subscribe. Because
successful subscriptions are the minority outcome, the evaluation
considers precision, recall, and F1-score in addition to accuracy.



## Dataset

* **Source:** [UCI Machine Learning Repository --- Bank
Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* **File used:** `bank-additional-full.csv`
* **Observations:** 41,188
* **Columns:** 21 (20 input variables and the target)
* **Target:** `y` --- whether the customer subscribed to a term
deposit (`yes` or `no`)



## Approach



### 1\. Data Understanding

The dataset was loaded with pandas and examined using DataFrame
inspection methods, including `head()`, `info()`, and checks for missing
values.

The dataset contained no null values. Some categorical features
contained the value `unknown`. These values were retained as their own
categories rather than being treated as nulls or replaced with
assumptions.



### 2\. Feature Engineering

The modeling exercise focused on the bank client information features:

* Age
* Job
* Marital status
* Education
* Credit in default
* Housing loan
* Personal loan

Categorical features were one-hot encoded, with one category dropped
from each feature to avoid redundant dummy variables. Age was retained
as a numerical feature.

The resulting feature matrix contained **41,188 rows and 28 columns**.
The target variable was encoded as a binary outcome.



### 3\. Train/Test Split

The data was split into training and testing sets using an 80/20 split.
Stratification was used to preserve the target-class proportions in both
sets.

A random state of `10` was used for reproducibility.

* Training observations: 32,950
* Testing observations: 8,238



### 4\. Baseline and Initial Model Comparison

A `DummyClassifier` using the most-frequent strategy established a
baseline. Since most customers did not subscribe, this classifier
predicted `no` for every customer.

The initial models were:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Support Vector Machine (SVM)

The initial accuracy results were close to the baseline. However, the
confusion matrices and classification metrics showed that some models
predicted no positive subscriptions at all. This demonstrated why
accuracy alone is not sufficient for this imbalanced classification
problem.



### 5\. Model Improvement

Hyperparameter tuning was performed using `GridSearchCV`, with F1-score
as the selection metric. Class weighting was explored to address the
imbalance between customers who subscribed and those who did not.

The Logistic Regression, Decision Tree, and SVM models benefited from
balanced class weights. KNN was tuned using its neighborhood, weighting,
and distance parameters.



## Results

The following table summarizes the final test-set results recorded in
the notebook.

Model                             Accuracy   Precision   Recall   F1-score

\---

Baseline                            88.74%       0.00%    0.00%     0.0000
Original Logistic Regression        88.74%       0.00%    0.00%     0.0000
Original KNN                        87.58%      29.07%    7.11%     0.1143
Original Decision Tree              86.94%      28.24%   10.34%     0.1514
Original SVM                        88.74%       0.00%    0.00%     0.0000
Optimized Logistic Regression       58.72%      15.81%   61.64%     0.2517
Optimized KNN                       86.96%      27.74%    9.81%     0.1449
Optimized Decision Tree             58.78%      15.70%   60.88%     0.2497
Optimized SVM                       60.00%      16.35%   61.96%     0.2587

## 

## Key Findings

1. **Accuracy alone was misleading.** The baseline achieved 88.74%
accuracy by predicting that every customer would not subscribe.
Despite its high accuracy, it failed to identify any subscribers.
2. **Class weighting improved subscriber detection.** The optimized
Logistic Regression, Decision Tree, and SVM models achieved
substantially higher recall than their original versions. Their
accuracy decreased because they began identifying more customers in
the minority `yes` class, which also increased false-positive
predictions.
3. **The optimized SVM achieved the highest F1-score.** Its F1-score
was 0.2587, with 61.96% recall and 16.35% precision. The optimized
Logistic Regression and Decision Tree models were close behind.
4. **KNN favored a different trade-off.** It retained higher accuracy
and precision than the other optimized models, but its recall was
much lower. It identified a smaller share of actual subscribers.
5. **There is no single best model for every business objective.** A
bank seeking to identify a larger share of potential subscribers may
prioritize recall. A bank seeking to limit unnecessary follow-up
calls may place more emphasis on precision. The appropriate model
depends on the relative costs of missed subscribers and unsuccessful
contacts.



## Conclusion

This project demonstrated the importance of selecting evaluation metrics
that reflect the business problem. In an imbalanced classification task,
a model can achieve high accuracy while failing to identify the outcome
the business actually cares about.

Among the models evaluated, the optimized SVM produced the highest
F1-score, while optimized Logistic Regression and Decision Tree models
delivered similar results. However, all optimized models had relatively
low precision, meaning many customers predicted as likely subscribers
did not actually subscribe.

The results are useful as a learning exercise and a starting point for
further investigation, but additional work would be needed before
deployment. Potential next steps include evaluating probability
thresholds against business costs, examining feature importance and
model interpretability, testing additional imbalance-handling
approaches, and validating performance on newer campaign data.



## Tools and Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab



## Project Files

* `Practical Application 3 - Final Draft.ipynb` --- notebook containing the analysis, model
training, tuning, and evaluation.
* `bank-additional-full.csv` --- dataset used in the project.

