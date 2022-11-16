#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Classification
# 
# Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this Challenge, you’ll use various techniques to train and evaluate models with imbalanced classes. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.
# 
# ## Instructions:
# 
# This challenge consists of the following subsections:
# 
# * Split the Data into Training and Testing Sets
# 
# * Create a Logistic Regression Model with the Original Data
# 
# * Predict a Logistic Regression Model with Resampled Training Data 
# 
# ### Split the Data into Training and Testing Sets
# 
# Open the starter code notebook and then use it to complete the following steps.
# 
# 1. Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.
# 
# 2. Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns.
# 
#     > **Note** A value of `0` in the “loan_status” column means that the loan is healthy. A value of `1` means that the loan has a high risk of defaulting.  
# 
# 3. Check the balance of the labels variable (`y`) by using the `value_counts` function.
# 
# 4. Split the data into training and testing datasets by using `train_test_split`.
# 
# ### Create a Logistic Regression Model with the Original Data
# 
# Employ your knowledge of logistic regression to complete the following steps:
# 
# 1. Fit a logistic regression model by using the training data (`X_train` and `y_train`).
# 
# 2. Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.
# 
# 3. Evaluate the model’s performance by doing the following:
# 
#     * Calculate the accuracy score of the model.
# 
#     * Generate a confusion matrix.
# 
#     * Print the classification report.
# 
# 4. Answer the following question: How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?
# 
# ### Predict a Logistic Regression Model with Resampled Training Data
# 
# Did you notice the small number of high-risk loan labels? Perhaps, a model that uses resampled data will perform better. You’ll thus resample the training data and then reevaluate the model. Specifically, you’ll use `RandomOverSampler`.
# 
# To do so, complete the following steps:
# 
# 1. Use the `RandomOverSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 
# 
# 2. Use the `LogisticRegression` classifier and the resampled data to fit the model and make predictions.
# 
# 3. Evaluate the model’s performance by doing the following:
# 
#     * Calculate the accuracy score of the model.
# 
#     * Generate a confusion matrix.
# 
#     * Print the classification report.
#     
# 4. Answer the following question: How well does the logistic regression model, fit with oversampled data, predict both the `0` (healthy loan) and `1` (high-risk loan) labels?
# 
# ### Write a Credit Risk Analysis Report
# 
# For this section, you’ll write a brief report that includes a summary and an analysis of the performance of both machine learning models that you used in this challenge. You should write this report as the `README.md` file included in your GitHub repository.
# 
# Structure your report by using the report template that `Starter_Code.zip` includes, and make sure that it contains the following:
# 
# 1. An overview of the analysis: Explain the purpose of this analysis.
# 
# 
# 2. The results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of both machine learning models.
# 
# 3. A summary: Summarize the results from the machine learning models. Compare the two versions of the dataset predictions. Include your recommendation for the model to use, if any, on the original vs. the resampled data. If you don’t recommend either model, justify your reasoning.

# In[ ]:


# Import the modules
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# ---

# ## Split the Data into Training and Testing Sets

# ### Step 1: Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.

# In[ ]:


# Read the CSV file from the Resources folder into a Pandas DataFrame
lending_data_df = pd.read_csv(
    Path("./Resources/lending_data.csv")
)

# Review the DataFrame
print("lending_data.csv file read into DataFrame:")
display(lending_data_df.head())


# ### Step 2: Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns.

# In[ ]:


# Separate the data into labels and features

# Separate the y variable, the labels
y_srs = lending_data_df['loan_status']

# Separate the X variable, the features
X_df = lending_data_df.drop(columns='loan_status')


# In[ ]:


# Review the y variable Series
print("Target set to be used for ML models: loan_status")
display(type(y_srs))
display(y_srs)


# In[ ]:


# Review the X variable DataFrame
print("Feature set DataFrame be used for ML models")
display(type(X_df))
display(X_df)


# ### Step 3: Check the balance of the labels variable (`y`) by using the `value_counts` function.

# In[ ]:


# Check the balance of our target values
print("# of loan_status values that =0 (loan approved) and =1 (loan rejected)")
display(y_srs.value_counts())


# ### Step 4: Split the data into training and testing datasets by using `train_test_split`.

# In[ ]:


# Split the data using train_test_split
# Assign a random_state of 1 to the function
X_train_df, X_test_df, y_train_srs, y_test_srs = train_test_split(X_df, y_srs, random_state=1)


# In[ ]:


# Data checkpoint
print("# of loan_status values that =0 (loan approved) and =1 (loan rejected) in the Training Set")
display(y_train_srs.value_counts())
print("# of loan_status values that =0 (loan approved) and =1 (loan rejected) in the Target/Test Set")
display(y_test_srs.value_counts())


# ---

# ## Create Regression Models with the Original Data

# ###  Step 1a: Fit a logistic regression model by using the training data (`X_train` and `y_train`).

# In[ ]:


# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
logreg_model = LogisticRegression(random_state=1)

# Fit the model using training data
logreg_model.fit(X_train_df, y_train_srs)


# ###  Step 1b: Fit a Support Vector Machine model by using the training data (`X_train` and `y_train`).

# In[ ]:


#### svm_model = SVC(kernel='linear')
#### 
#### # Fit the data
#### svm_model.fit(X_train_df, y_train_srs)


# ###  Step 1c: Fit a Decision Tree model by using the training data (`X_train` and `y_train`).

# In[ ]:


# Creating the decision tree classifier instance
dectree_model = tree.DecisionTreeClassifier(random_state=1)

# Fitting the model
dectree_model = dectree_model.fit(X_train_df, y_train_srs)


# ###  Step 1d: Fit a Random Forest model by using the training data (`X_train` and `y_train`).

# In[ ]:


# Create a random forest classifier
rndfor_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fitting the model
rndfor_model = rndfor_model.fit(X_train_df, y_train_srs)


# ###  Step 1e: Fit a KNN model by using the training data (`X_train` and `y_train`).

# In[ ]:


# Instantiate the model with k = 3 neighbors
knn_model = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn_model.fit(X_train_df, y_train_srs)


# ### Step 2: Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.

# In[ ]:


# Make a prediction using the testing data on the logistics Regression model
logreg_y_pred_npa = logreg_model.predict(X_test_df)

#### # Make a prediction using the testing data on the Support Vector Matrix model
#### svm_y_pred_npa = svm_model.predict(X_test_df)

# Make a prediction using the testing data on the Decision Tree model
dectree_y_pred_npa = dectree_model.predict(X_test_df)

# Make a prediction using the testing data on the Random Forest model
rndfor_y_pred_npa = rndfor_model.predict(X_test_df)

# Make a prediction using the testing data on the KNN model
knn_y_pred_npa = knn_model.predict(X_test_df)


# ### Step 3: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores:")
print("-------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_npa))


# In[ ]:


# Print confusion matrix
def confusion_matrix_sklearn(y_test, y_pred, plt_title):
    """
    To plot the confusion_matrix with percentages
    prediction:  predicted values
    original:    original values
    """
    cm = confusion_matrix(y_test, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(3, 2))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(plt_title)


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_npa, "Logistic Regression")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_npa, "SVM")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_npa, "Decision Tree")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_npa, "Random Forest")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_npa, "KNN")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_class_rpt = classification_report(y_test_srs, logreg_y_pred_npa)

#### # Print the classification report for the SVM model
#### svm_class_rpt = classification_report(y_test_srs, svm_y_pred_npa)

# Print the classification report for the Decision Tree model
dectree_class_rpt = classification_report(y_test_srs, dectree_y_pred_npa)

# Print the classification report for the Random Forest model
rndfor_class_rpt = classification_report(y_test_srs, rndfor_y_pred_npa)

# Print the classification report for the KNN model
knn_class_rpt = classification_report(y_test_srs, knn_y_pred_npa)


# In[ ]:


print("Logistic Regression Classification Report:\n", logreg_class_rpt)
####print("SVM Classification Report:\n", svm_class_rpt)
print("Decision Tree Classification Report:\n", dectree_class_rpt)
print("Random Forest Classification Report:\n", rndfor_class_rpt)
print("KNN Classification Report:\n", knn_class_rpt)


# ### Step 4: Answer the following question.

# **Question:** How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?
# 
# **Answer:** \
# - Logistics regression classification report analysis:\
# -- "accuracy = 0.99".  The model predicted correctly 99% of the time that the loan was either healhty loan or a high risk loan\
# -- "precision /0 (healthy loan) = 1.00".  Out of all the loans the model predicted would be a healthy loan (0), the model predicted this correctly 100% of the time\
# -- "precision /1 (high-risk loan) = 0.84".  Out of all the loans the model predicted would be a high-risk loan (1), the model predicted this correctly 84% of the time\
# -- "recall /0 (healthy loan) = 0.99".  Out of all the loans that were declared a healthy loan, the model predicted this outcome 99% of the time\
# -- "recall /1 (high-risk loan) = 0.85".  Out of all the loans that were declared a high-risk loan, the model predicted this outcome 85% of the time

# ---

# ## Predict a Logistic Regression Model with Resampled Training Data

# ### Step 1a: Use the `RandomOverSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

# In[ ]:


# Instantiate the random oversampler model
random_oversampler = RandomOverSampler(random_state=1)

# Fit the original training data to the random_oversampler model
X_train_rndovr_df, y_train_rndovr_srs = random_oversampler.fit_resample(X_train_df, y_train_srs)


# In[ ]:


# Count the distinct values of the resampled labels data
print("# of random over sampled loan_status values that =0 (loan approved) and =1 (loan rejected)")
display(y_train_rndovr_srs.value_counts())


# ### Step 2a-a: Use the `LogisticRegression` classifier and the oversampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
logreg_rndovr_model = LogisticRegression(random_state=1)

# Fit the model using the rndovr training data
logreg_rndovr_model.fit(X_train_rndovr_df, y_train_rndovr_srs)

# Make a prediction using the testing data
logreg_y_pred_rndovr_npa = logreg_rndovr_model.predict(X_test_df)


# ### Step 2a-b: Use the `SVM` classifier and the oversampled resampled data to fit the model and make predictions.

# #### (NOTE: THE SVM MODEL TAKES A VERY LONG TIME TO RUN, EXCLUDING IT FROM THE ANALYSIS - LEAVING IT IN AS COMMENTS FOR FUTURE REFERENCE)

# In[ ]:


#### # Instantiate the model
#### svm_rndovr_model = SVC(kernel='linear')
#### 
#### # Fit the model using the rndovr training data
#### svm_rndovr_model.fit(X_train_rndovr_df, y_train_rndovr_srs)
#### 
#### # Make a prediction using the testing data
#### svm_y_pred_rndovr_npa = svm_rndovr_model.predict(X_test_df)


# ### Step 2a-c: Use the `Decision Tree` classifier and the oversampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Creating the decision tree classifier instance
dectree_rndovr_model = tree.DecisionTreeClassifier(random_state=1)

# Fit the model using the rndovr training data
dectree_rndovr_model.fit(X_train_rndovr_df, y_train_rndovr_srs)

# Make a prediction using the testing data
dectree_y_pred_rndovr_npa = dectree_rndovr_model.predict(X_test_df)


# ### Step 2a-d: Use the `Random Forest` classifier and the oversampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Create a random forest classifier
rndfor_rndovr_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fit the model using the rndovr training data
rndfor_rndovr_model.fit(X_train_rndovr_df, y_train_rndovr_srs)

# Make a prediction using the testing data
rndfor_y_pred_rndovr_npa = rndfor_rndovr_model.predict(X_test_df)


# ### Step 2a-e: Use the `KNN` classifier and the oversampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Instantiate the model with k = 3 neighbors
knn_rndovr_model = KNeighborsClassifier(n_neighbors=3)

# Fit the model using the rndovr training data
knn_rndovr_model.fit(X_train_rndovr_df, y_train_rndovr_srs)

# Make a prediction using the testing data
knn_y_pred_rndovr_npa = knn_rndovr_model.predict(X_test_df)


# ### Step 3a: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores (Random Oversampled):")
print("----------------------------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_rndovr_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_rndovr_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_rndovr_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_rndovr_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_rndovr_npa))


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_rndovr_npa, "Logistic Regression w/ Rand Oversampling")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_rndovr_npa, "SVM w/ Rand Oversampling")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_rndovr_npa, "Decision Tree w/ Rand Oversampling")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_rndovr_npa, "Random Forest w/ Rand Oversampling")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_rndovr_npa, "KNN w/ Rand Oversampling")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_rndovr_class_rpt = classification_report(y_test_srs, logreg_y_pred_rndovr_npa)

#### # Print the classification report for the SVM model
#### svm_rndovr_class_rpt = classification_report(y_test_srs, svm_y_pred_rndovr_npa)

# Print the classification report for the Decision Tree model
dectree_rndovr_class_rpt = classification_report(y_test_srs, dectree_y_pred_rndovr_npa)

# Print the classification report for the Random Forest model
rndfor_rndovr_class_rpt = classification_report(y_test_srs, rndfor_y_pred_rndovr_npa)

# Print the classification report for the KNN model
knn_rndovr_class_rpt = classification_report(y_test_srs, knn_y_pred_rndovr_npa)


# In[ ]:


print("Logistic Regression (w/ Random Oversampling) Classification Report:\n", logreg_rndovr_class_rpt)
####print("SVM (ww/ Random Oversampling) Classification Report:\n", svm_rndovr_class_rpt)
print("Decision Tree (ww/ Random Oversampling) Classification Report:\n", dectree_rndovr_class_rpt)
print("Random Forest (ww/ Random Oversampling) Classification Report:\n", rndfor_rndovr_class_rpt)
print("KNN (ww/ Random Oversampling) Classification Report:\n", knn_rndovr_class_rpt)


# ### Step 4: Answer the following question

# **Question:** How well does the logistic regression model, fit with oversampled data, predict both the `0` (healthy loan) and `1` (high-risk loan) labels?
# 
# **Answer:** \
# - Logistics Regression (w/ Random Oversampling) classification report analysis:\
# -- "accuracy = 0.99".  The model predicted correctly 99% of the time that the loan was either healhty loan or a high risk loan\
#                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;** No improvement using random oversampling compared to the original non oversampling accuracy score of 0.99 **\
# -- "precision /0 (healthy loan) = 1.00".  Out of all the loans the model predicted would be a healthy loan (0), the model predicted this correctly 100% of the time\
#                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;** No improvement using random oversampling compared to the original non oversampling precision /0 score of 1.00**\
# -- "precision /1 (high-risk loan) = 0.84".  Out of all the loans the model predicted would be a high-risk loan (1), the model predicted this correctly 84% of the time\
#                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;** No improvement using random oversampling compared to the original non oversampling precision /1 score of 0.84 **\
# -- "recall /0 (healthy loan) = 0.99".  Out of all the loans that were declared a healthy loan, the model predicted this outcome 99% of the time\
#                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;** No improvement using random oversampling compared to the original non oversampling recall /0 score of 0.99**\
# -- "recall /1 (high-risk loan) = 0.99".  Out of all the loans that were declared a high-risk loan, the model predicted this outcome 99% of the time
#                        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;** Here we do have a considerable improvement using random oversampling compared to the original non oversampling recall /1 score of 0.85**\

# ### Step 1b: Use the `RandomUnderSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

# In[ ]:


# Instantiate the random undersampler model
random_undersampler = RandomUnderSampler(random_state=1)

# Fit the original training data to the random_undersampler model
X_train_rndunder_df, y_train_rndunder_srs = random_undersampler.fit_resample(X_train_df, y_train_srs)


# In[ ]:


# Count the distinct values of the resampled labels data
print("# of random under sampled loan_status values that =0 (loan approved) and =1 (loan rejected)")
display(y_train_rndunder_srs.value_counts())


# ### Step 2b-a: Use the `LogisticRegression` classifier and the undersampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
logreg_rndunder_model = LogisticRegression(random_state=1)

# Fit the model using the rndunder training data
logreg_rndunder_model.fit(X_train_rndunder_df, y_train_rndunder_srs)

# Make a prediction using the testing data
logreg_y_pred_rndunder_npa = logreg_rndunder_model.predict(X_test_df)


# ### Step 2b-b: Use the `SVM` classifier and the undersampled resampled data to fit the model and make predictions.

# In[ ]:


#### # Instantiate the model
#### svm_rndunder_model = SVC(kernel='linear')
#### 
#### # Fit the model using the rndunder training data
#### svm_rndunder_model.fit(X_train_rndunder_df, y_train_rndunder_srs)
#### 
#### # Make a prediction using the testing data
#### svm_y_pred_rndunder_npa = svm_rndunder_model.predict(X_test_df)


# ### Step 2b-c: Use the `Decision Tree` classifier and the undersampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Creating the decision tree classifier instance
dectree_rndunder_model = tree.DecisionTreeClassifier(random_state=1)

# Fit the model using the rndunder training data
dectree_rndunder_model.fit(X_train_rndunder_df, y_train_rndunder_srs)

# Make a prediction using the testing data
dectree_y_pred_rndunder_npa = dectree_rndunder_model.predict(X_test_df)


# ### Step 2b-d: Use the `Random Forest` classifier and the undersampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Create a random forest classifier
rndfor_rndunder_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fit the model using the rndunder training data
rndfor_rndunder_model.fit(X_train_rndunder_df, y_train_rndunder_srs)

# Make a prediction using the testing data
rndfor_y_pred_rndunder_npa = rndfor_rndunder_model.predict(X_test_df)


# ### Step 2b-e: Use the `KNN` classifier and the undersampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Instantiate the model with k = 3 neighbors
knn_rndunder_model = KNeighborsClassifier(n_neighbors=3)

# Fit the model using the rndunder training data
knn_rndunder_model.fit(X_train_rndunder_df, y_train_rndunder_srs)

# Make a prediction using the testing data
knn_y_pred_rndunder_npa = knn_rndunder_model.predict(X_test_df)


# ### Step 3b: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores (Random Undersampled):")
print("-----------------------------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_rndunder_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_rndunder_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_rndunder_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_rndunder_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_rndunder_npa))


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_rndunder_npa, "Logistic Regression w/ Rand Undersampling")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_rndunder_npa, "SVM w/ Rand Undersampling")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_rndunder_npa, "Decision Tree w/ Rand Undersampling")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_rndunder_npa, "Random Forest w/ Rand Undersampling")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_rndunder_npa, "KNN w/ Rand Undersampling")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_rndunder_class_rpt = classification_report(y_test_srs, logreg_y_pred_rndunder_npa)

#### # Print the classification report for the SVM model
#### svm_rndunder_class_rpt = classification_report(y_test_srs, svm_y_pred_rndunder_npa)

# Print the classification report for the Decision Tree model
dectree_rndunder_class_rpt = classification_report(y_test_srs, dectree_y_pred_rndunder_npa)

# Print the classification report for the Random Forest model
rndfor_rndunder_class_rpt = classification_report(y_test_srs, rndfor_y_pred_rndunder_npa)

# Print the classification report for the KNN model
knn_rndunder_class_rpt = classification_report(y_test_srs, knn_y_pred_rndunder_npa)


# In[ ]:


print("Logistic Regression (w/ Random Undersampling) Classification Report:\n", logreg_rndunder_class_rpt)
####print("SVM (ww/ Random Undersampling) Classification Report:\n", svm_rndunder_class_rpt)
print("Decision Tree (ww/ Random Undersampling) Classification Report:\n", dectree_rndunder_class_rpt)
print("Random Forest (ww/ Random Undersampling) Classification Report:\n", rndfor_rndunder_class_rpt)
print("KNN (ww/ Random Undersampling) Classification Report:\n", knn_rndunder_class_rpt)


# ### Step 1c: Use the `SMOTE Oversampling` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

# In[ ]:


# Instantiate a SMOTE instance
smote_sampler =  SMOTE(random_state=1, sampling_strategy='auto')
# Fit the training data to the SMOTE model
X_train_smote_df, y_train_smote_srs = smote_sampler.fit_resample(X_train_df, y_train_srs)
# Count distinct values for the resampled target data


# In[ ]:


# Count the distinct values of the resampled labels data
print("# of SMOTE Sampled loan_status values that =0 (loan approved) and =1 (loan rejected)")
display(y_train_smote_srs.value_counts())


# ### Step 2c-a: Use the `LogisticRegression` classifier and the smote sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
logreg_smote_model = LogisticRegression(random_state=1)

# Fit the model using the smote training data
logreg_smote_model.fit(X_train_smote_df, y_train_smote_srs)

# Make a prediction using the testing data
logreg_y_pred_smote_npa = logreg_smote_model.predict(X_test_df)


# ### Step 2c-b: Use the `SVM` classifier and the smote sampled data to fit the model and make predictions.

# In[ ]:


#### # Instantiate the model
#### svm_smote_model = SVC(kernel='linear')
#### 
#### # Fit the model using the smote training data
#### svm_smote_model.fit(X_train_smote_df, y_train_smote_srs)
#### 
#### # Make a prediction using the testing data
#### svm_y_pred_smote_npa = svm_smote_model.predict(X_test_df)


# ### Step 2c-c: Use the `Decision Tree` classifier and the smote sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Creating the decision tree classifier instance
dectree_smote_model = tree.DecisionTreeClassifier(random_state=1)

# Fit the model using the smote training data
dectree_smote_model.fit(X_train_smote_df, y_train_smote_srs)

# Make a prediction using the testing data
dectree_y_pred_smote_npa = dectree_smote_model.predict(X_test_df)


# ### Step 2c-d: Use the `Random Forest` classifier and the smote sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Create a random forest classifier
rndfor_smote_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fit the model using the smote training data
rndfor_smote_model.fit(X_train_smote_df, y_train_smote_srs)

# Make a prediction using the testing data
rndfor_y_pred_smote_npa = rndfor_smote_model.predict(X_test_df)


# ### Step 2c-e: Use the `KNN` classifier and the smote sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Instantiate the model with k = 3 neighbors
knn_smote_model = KNeighborsClassifier(n_neighbors=3)

# Fit the model using the smote training data
knn_smote_model.fit(X_train_smote_df, y_train_smote_srs)

# Make a prediction using the testing data
knn_y_pred_smote_npa = knn_smote_model.predict(X_test_df)


# ### Step 3c: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores (SMOTE Sampled):")
print("-----------------------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_smote_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_smote_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_smote_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_smote_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_smote_npa))


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_smote_npa, "Logistic Regression w/ SMOTE Sampled")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_smote_npa, "SVM w/ SMOTE Sampled")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_smote_npa, "Decision Tree w/ SMOTE Sampled")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_smote_npa, "Random Forest w/ SMOTE Sampled")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_smote_npa, "KNN w/ SMOTE Sampled")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_smote_class_rpt = classification_report(y_test_srs, logreg_y_pred_smote_npa)

#### # Print the classification report for the SVM model
#### svm_smote_class_rpt = classification_report(y_test_srs, svm_y_pred_smote_npa)

# Print the classification report for the Decision Tree model
dectree_smote_class_rpt = classification_report(y_test_srs, dectree_y_pred_smote_npa)

# Print the classification report for the Random Forest model
rndfor_smote_class_rpt = classification_report(y_test_srs, rndfor_y_pred_smote_npa)

# Print the classification report for the KNN model
knn_smote_class_rpt = classification_report(y_test_srs, knn_y_pred_smote_npa)


# In[ ]:


print("Logistic Regression (w/ SMOTE Sampled) Classification Report:\n", logreg_smote_class_rpt)
####print("SVM (ww/ SMOTE Sampled) Classification Report:\n", svm_smote_class_rpt)
print("Decision Tree (ww/ SMOTE Sampled)g Classification Report:\n", dectree_smote_class_rpt)
print("Random Forest (ww/ SMOTE Sampled)g Classification Report:\n", rndfor_smote_class_rpt)
print("KNN (ww/ SMOTE Sampled) Classification Report:\n", knn_smote_class_rpt)


# ### Step 1d: Use the `SMOTEENN Combined Sampling` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

# In[ ]:


# Instantiate a SMOTEENN instance
smoteenn_sampler =  SMOTEENN(random_state=1)
# Fit the training data to the SMOTEENN model
X_train_smoteenn_df, y_train_smoteenn_srs = smoteenn_sampler.fit_resample(X_train_df, y_train_srs)
# Count distinct values for the resampled target data


# In[ ]:


# Count the distinct values of the resampled labels data
print("# of SMOTEENN Sampled loan_status values that =0 (loan approved) and =1 (loan rejected)")
display(y_train_smoteenn_srs.value_counts())


# ### Step 2d-a: Use the `LogisticRegression` classifier and the smoteenn sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
logreg_smoteenn_model = LogisticRegression(random_state=1)

# Fit the model using the smoteenn training data
logreg_smoteenn_model.fit(X_train_smoteenn_df, y_train_smoteenn_srs)

# Make a prediction using the testing data
logreg_y_pred_smoteenn_npa = logreg_smoteenn_model.predict(X_test_df)


# ### Step 2d-b: Use the `SVM` classifier and the smoteenn sampled data to fit the model and make predictions.

# In[ ]:


#### # Instantiate the model
#### svm_smoteenn_model = SVC(kernel='linear')
#### 
#### # Fit the model using the smoteenn training data
#### svm_smoteenn_model.fit(X_train_smoteenn_df, y_train_smoteenn_srs)
#### 
#### # Make a prediction using the testing data
#### svm_y_pred_smoteenn_npa = svm_smoteenn_model.predict(X_test_df)


# ### Step 2d-c: Use the `Decision Tree` classifier and the smoteenn sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Creating the decision tree classifier instance
dectree_smoteenn_model = tree.DecisionTreeClassifier(random_state=1)

# Fit the model using the smoteenn training data
dectree_smoteenn_model.fit(X_train_smoteenn_df, y_train_smoteenn_srs)

# Make a prediction using the testing data
dectree_y_pred_smoteenn_npa = dectree_smoteenn_model.predict(X_test_df)


# ### Step 2d-d: Use the `Random Forest` classifier and the smoteenn sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Create a random forest classifier
rndfor_smoteenn_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fit the model using the smoteenn training data
rndfor_smoteenn_model.fit(X_train_smoteenn_df, y_train_smoteenn_srs)

# Make a prediction using the testing data
rndfor_y_pred_smoteenn_npa = rndfor_smoteenn_model.predict(X_test_df)


# ### Step 2d-e: Use the `KNN` classifier and the smoteenn sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Instantiate the model with k = 3 neighbors
knn_smoteenn_model = KNeighborsClassifier(n_neighbors=3)

# Fit the model using the smoteenn training data
knn_smoteenn_model.fit(X_train_smoteenn_df, y_train_smoteenn_srs)

# Make a prediction using the testing data
knn_y_pred_smoteenn_npa = knn_smoteenn_model.predict(X_test_df)


# ### Step 3d: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores (SMOTEENN Sampled):")
print("--------------------------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_smoteenn_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_smoteenn_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_smoteenn_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_smoteenn_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_smoteenn_npa))


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_smoteenn_npa, "Logistic Regression w/ SMOTEENN Sampled")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_smoteenn_npa, "SVM w/ SMOTEENN Sampled")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_smoteenn_npa, "Decision Tree w/ SMOTEENN Sampled")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_smoteenn_npa, "Random Forest w/ SMOTEENN Sampled")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_smoteenn_npa, "KNN w/ SMOTEENN Sampled")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_smoteenn_class_rpt = classification_report(y_test_srs, logreg_y_pred_smoteenn_npa)

#### # Print the classification report for the SVM model
#### svm_smoteenn_class_rpt = classification_report(y_test_srs, svm_y_pred_smoteenn_npa)

# Print the classification report for the Decision Tree model
dectree_smoteenn_class_rpt = classification_report(y_test_srs, dectree_y_pred_smoteenn_npa)

# Print the classification report for the Random Forest model
rndfor_smoteenn_class_rpt = classification_report(y_test_srs, rndfor_y_pred_smoteenn_npa)

# Print the classification report for the KNN model
knn_smoteenn_class_rpt = classification_report(y_test_srs, knn_y_pred_smoteenn_npa)


# In[ ]:


print("Logistic Regression (w/ SMOTEENN Sampled) Classification Report:\n", logreg_smoteenn_class_rpt)
####print("SVM (ww/ SMOTEENN Sampled) Classification Report:\n", svm_smoteenn_class_rpt)
print("Decision Tree (ww/ SMOTEENN Sampled)g Classification Report:\n", dectree_smoteenn_class_rpt)
print("Random Forest (ww/ SMOTEENN Sampled)g Classification Report:\n", rndfor_smoteenn_class_rpt)
print("KNN (ww/ SMOTEENN Sampled) Classification Report:\n", knn_smoteenn_class_rpt)


# ### Step e: Use the `ClusterCentroids` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

# In[ ]:


# Instantiate a ClusterCentroids instance
cluster_centroid_sampler = ClusterCentroids(random_state=1)
# Fit the training data to the cluster centroids model
X_train_clstrcntrd_df, y_train_clstrcntrd_srs = cluster_centroid_sampler.fit_resample(X_train_df, y_train_srs)
# Count distinct values for the resampled target data


# In[ ]:


# Count the distinct values of the resampled labels data
print("# of cluster centroid loan_status values that =0 (loan approved) and =1 (loan rejected)")
display(y_train_clstrcntrd_srs.value_counts())


# ### Step 2e-a: Use the `LogisticRegression` classifier and the cluster centroid sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
logreg_clstrcntrd_model = LogisticRegression(random_state=1)

# Fit the model using the clstrcntrd training data
logreg_clstrcntrd_model.fit(X_train_clstrcntrd_df, y_train_clstrcntrd_srs)

# Make a prediction using the testing data
logreg_y_pred_clstrcntrd_npa = logreg_clstrcntrd_model.predict(X_test_df)


# ### Step 2e-b: Use the `SVM` classifier and the cluster centroid sampled data to fit the model and make predictions.

# In[ ]:


#### # Instantiate the model
#### svm_clstrcntrd_model = SVC(kernel='linear')
#### 
#### # Fit the model using the clstrcntrd training data
#### svm_clstrcntrd_model.fit(X_train_clstrcntrd_df, y_train_clstrcntrd_srs)
#### 
#### # Make a prediction using the testing data
#### svm_y_pred_clstrcntrd_npa = svm_clstrcntrd_model.predict(X_test_df)


# ### Step 2e-c: Use the `Decision Tree` classifier and the cluster centroid sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Creating the decision tree classifier instance
dectree_clstrcntrd_model = tree.DecisionTreeClassifier(random_state=1)

# Fit the model using the clstrcntrd training data
dectree_clstrcntrd_model.fit(X_train_clstrcntrd_df, y_train_clstrcntrd_srs)

# Make a prediction using the testing data
dectree_y_pred_clstrcntrd_npa = dectree_clstrcntrd_model.predict(X_test_df)


# ### Step 2e-d: Use the `Random Forest` classifier and the cluster centroid sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Create a random forest classifier
rndfor_clstrcntrd_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fit the model using the clstrcntrd training data
rndfor_clstrcntrd_model.fit(X_train_clstrcntrd_df, y_train_clstrcntrd_srs)

# Make a prediction using the testing data
rndfor_y_pred_clstrcntrd_npa = rndfor_clstrcntrd_model.predict(X_test_df)


# ### Step 2e-e: Use the `KNN` classifier and the cluster centroid sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Instantiate the model with k = 3 neighbors
knn_clstrcntrd_model = KNeighborsClassifier(n_neighbors=3)

# Fit the model using the clstrcntrd training data
knn_clstrcntrd_model.fit(X_train_clstrcntrd_df, y_train_clstrcntrd_srs)

# Make a prediction using the testing data
knn_y_pred_clstrcntrd_npa = knn_clstrcntrd_model.predict(X_test_df)


# ### Step 3e: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores (Cluster Centroid Sampled):")
print("----------------------------------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_clstrcntrd_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_clstrcntrd_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_clstrcntrd_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_clstrcntrd_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_clstrcntrd_npa))


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_clstrcntrd_npa, "Logistic Regression w/ Cluster Centroid Sampled")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_clstrcntrd_npa, "SVM w/ Cluster Centroid Sampled")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_clstrcntrd_npa, "Decision Tree w/ Cluster Centroid Sampled")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_clstrcntrd_npa, "Random Forest w/ Cluster Centroid Sampled")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_clstrcntrd_npa, "KNN w/ Cluster Centroid Sampled")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_clstrcntrd_class_rpt = classification_report(y_test_srs, logreg_y_pred_clstrcntrd_npa)

#### # Print the classification report for the SVM model
#### svm_clstrcntrd_class_rpt = classification_report(y_test_srs, svm_y_pred_clstrcntrd_npa)

# Print the classification report for the Decision Tree model
dectree_clstrcntrd_class_rpt = classification_report(y_test_srs, dectree_y_pred_clstrcntrd_npa)

# Print the classification report for the Random Forest model
rndfor_clstrcntrd_class_rpt = classification_report(y_test_srs, rndfor_y_pred_clstrcntrd_npa)

# Print the classification report for the KNN model
knn_clstrcntrd_class_rpt = classification_report(y_test_srs, knn_y_pred_clstrcntrd_npa)


# In[ ]:


print("Logistic Regression (w/ Cluster Centroid Sampled) Classification Report:\n", logreg_clstrcntrd_class_rpt)
####print("SVM (ww/ Cluster Centroid Sampled) Classification Report:\n", svm_clstrcntrd_class_rpt)
print("Decision Tree (ww/ Cluster Centroid Sampled)g Classification Report:\n", dectree_clstrcntrd_class_rpt)
print("Random Forest (ww/ Cluster Centroid Sampled)g Classification Report:\n", rndfor_clstrcntrd_class_rpt)
print("KNN (ww/ Cluster Centroid Sampled) Classification Report:\n", knn_clstrcntrd_class_rpt)


# In[ ]:


# (Re-)Calculate classificaiton report to output to a dictionary format

# Logistic Regression
logreg_class_rpt_dict = classification_report(y_test_srs, logreg_y_pred_npa, output_dict = True)
#### svm_class_rpt_dict = classification_report(y_test_srs, svm_y_pred_npa, output_dict = True)
dectree_class_rpt_dict = classification_report(y_test_srs, dectree_y_pred_npa, output_dict = True)
rndfor_class_rpt_dict = classification_report(y_test_srs, rndfor_y_pred_npa, output_dict = True)
knn_class_rpt_dict = classification_report(y_test_srs, knn_y_pred_npa, output_dict = True)

logreg_rndovr_class_rpt_dict = classification_report(y_test_srs, logreg_y_pred_rndovr_npa, output_dict = True)
#### svm_rndovr_class_rpt_dict = classification_report(y_test_srs, svm_y_pred_rndovr_np, output_dict = Truea)
dectree_rndovr_class_rpt_dict = classification_report(y_test_srs, dectree_y_pred_rndovr_npa, output_dict = True)
rndfor_rndovr_class_rpt_dict = classification_report(y_test_srs, rndfor_y_pred_rndovr_npa, output_dict = True)
knn_rndovr_class_rpt_dict = classification_report(y_test_srs, knn_y_pred_rndovr_npa, output_dict = True)

logreg_rndunder_class_rpt_dict = classification_report(y_test_srs, logreg_y_pred_rndunder_npa, output_dict = True)
#### svm_rndunder_class_rpt_dict = classification_report(y_test_srs, svm_y_pred_rndunder_npa, output_dict = True)
dectree_rndunder_class_rpt_dict = classification_report(y_test_srs, dectree_y_pred_rndunder_npa, output_dict = True)
rndfor_rndunder_class_rpt_dict = classification_report(y_test_srs, rndfor_y_pred_rndunder_npa, output_dict = True)
knn_rndunder_class_rpt_dict = classification_report(y_test_srs, knn_y_pred_rndunder_npa, output_dict = True)

logreg_smote_class_rpt_dict = classification_report(y_test_srs, logreg_y_pred_smote_npa, output_dict = True)
#### svm_smote_class_rpt_dict = classification_report(y_test_srs, svm_y_pred_smote_npa, output_dict = True)
dectree_smote_class_rpt_dict = classification_report(y_test_srs, dectree_y_pred_smote_npa, output_dict = True)
rndfor_smote_class_rpt_dict = classification_report(y_test_srs, rndfor_y_pred_smote_npa, output_dict = True)
knn_smote_class_rpt_dict = classification_report(y_test_srs, knn_y_pred_smote_npa, output_dict = True)

logreg_smoteenn_class_rpt_dict = classification_report(y_test_srs, logreg_y_pred_smoteenn_npa, output_dict = True)
#### svm_smoteenn_class_rpt_dict = classification_report(y_test_srs, svm_y_pred_smoteenn_npa, output_dict = True)
dectree_smoteenn_class_rpt_dict = classification_report(y_test_srs, dectree_y_pred_smoteenn_npa, output_dict = True)
rndfor_smoteenn_class_rpt_dict = classification_report(y_test_srs, rndfor_y_pred_smoteenn_npa, output_dict = True)
knn_smoteenn_class_rpt_dict = classification_report(y_test_srs, knn_y_pred_smoteenn_npa, output_dict = True)

logreg_clstrcntrd_class_rpt_dict = classification_report(y_test_srs, logreg_y_pred_clstrcntrd_npa, output_dict = True)
#### svm_clstrcntrd_class_rpt_dict = classification_report(y_test_srs, svm_y_pred_clstrcntrd_npa, output_dict = True)
dectree_clstrcntrd_class_rpt_dict = classification_report(y_test_srs, dectree_y_pred_clstrcntrd_npa, output_dict = True)
rndfor_clstrcntrd_class_rpt_dict = classification_report(y_test_srs, rndfor_y_pred_clstrcntrd_npa, output_dict = True)
knn_clstrcntrd_class_rpt_dict = classification_report(y_test_srs, knn_y_pred_clstrcntrd_npa, output_dict = True)


# ## Scale the data

# In[ ]:


# Scale the data
scaler = StandardScaler()
X_scaler = scaler.fit(X_train_df)
X_train_scaled_npa = X_scaler.transform(X_train_df)
X_test_scaled_npa = X_scaler.transform(X_test_df)


# ## Create Regression Models with the Original Data

# ###  Step 1a: Fit a logistic regression model by using the training data (`X_train` and `y_train`).

# In[ ]:


# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
logreg_model = LogisticRegression(random_state=1)

# Fit the model using training data
logreg_model.fit(X_train_scaled_npa, y_train_srs)


# ###  Step 1b: Fit a Support Vector Machine model by using the training data (`X_train` and `y_train`).

# In[ ]:


#### svm_model = SVC(kernel='linear')
#### 
#### # Fit the data
#### svm_model.fit(X_train_scaled_npa, y_train_srs)


# ###  Step 1c: Fit a Decision Tree model by using the training data (`X_train` and `y_train`).

# In[ ]:


# Creating the decision tree classifier instance
dectree_model = tree.DecisionTreeClassifier(random_state=1)

# Fitting the model
dectree_model = dectree_model.fit(X_train_scaled_npa, y_train_srs)


# ###  Step 1d: Fit a Random Forest model by using the training data (`X_train` and `y_train`).

# In[ ]:


# Create a random forest classifier
rndfor_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fitting the model
rndfor_model = rndfor_model.fit(X_train_scaled_npa, y_train_srs)


# ###  Step 1e: Fit a KNN model by using the training data (`X_train` and `y_train`).

# In[ ]:


# Instantiate the model with k = 3 neighbors
knn_model = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn_model.fit(X_train_scaled_npa, y_train_srs)


# ### Step 2: Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.

# In[ ]:


# Make a prediction using the testing data on the logistics Regression model
logreg_y_pred_npa = logreg_model.predict(X_test_scaled_npa)

#### # Make a prediction using the testing data on the Support Vector Matrix model
#### svm_y_pred_npa = svm_model.predict(X_test_scaled_npa)

# Make a prediction using the testing data on the Decision Tree model
dectree_y_pred_npa = dectree_model.predict(X_test_scaled_npa)

# Make a prediction using the testing data on the Random Forest model
rndfor_y_pred_npa = rndfor_model.predict(X_test_scaled_npa)

# Make a prediction using the testing data on the KNN model
knn_y_pred_npa = knn_model.predict(X_test_scaled_npa)


# ### Step 3: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores (w/ Scaling):")
print("--------------------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_npa))


# In[ ]:


# Print confusion matrix
def confusion_matrix_sklearn(y_test, y_pred, plt_title):
    """
    To plot the confusion_matrix with percentages
    prediction:  predicted values
    original:    original values
    """
    cm = confusion_matrix(y_test, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(3, 2))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(plt_title)


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_npa, "Logistic Regression (w/ Scaling)")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_npa, "SVM (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_npa, "Decision Tree (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_npa, "Random Forest (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_npa, "KNN (w/ Scaling)")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_class_rpt = classification_report(y_test_srs, logreg_y_pred_npa)

#### # Print the classification report for the SVM model
#### svm_class_rpt = classification_report(y_test_srs, svm_y_pred_npa)

# Print the classification report for the Decision Tree model
dectree_class_rpt = classification_report(y_test_srs, dectree_y_pred_npa)

# Print the classification report for the Random Forest model
rndfor_class_rpt = classification_report(y_test_srs, rndfor_y_pred_npa)

# Print the classification report for the KNN model
knn_class_rpt = classification_report(y_test_srs, knn_y_pred_npa)


# In[ ]:


print("Logistic Regression Classification Report (w/ Scaling):\n", logreg_class_rpt)
####print("SVM Classification Report (w/ Scaling):\n", svm_class_rpt)
print("Decision Tree Classification Report (w/ Scaling):\n", dectree_class_rpt)
print("Random Forest Classification Report (w/ Scaling):\n", rndfor_class_rpt)
print("KNN Classification Report (w/ Scaling):\n", knn_class_rpt)


# ---

# ## Predict a Logistic Regression Model with Resampled Training Data

# ### Step 1a: Use the `RandomOverSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

# In[ ]:


# Instantiate the random oversampler model
random_oversampler = RandomOverSampler(random_state=1)

# Fit the original training data to the random_oversampler model
X_train_rndovr_df, y_train_rndovr_srs = random_oversampler.fit_resample(X_train_scaled_npa, y_train_srs)


# In[ ]:


# Count the distinct values of the resampled labels data
print("# of random over sampled loan_status values that =0 (loan approved) and =1 (loan rejected)")
display(y_train_rndovr_srs.value_counts())


# ### Step 2a-a: Use the `LogisticRegression` classifier and the oversampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
logreg_rndovr_model = LogisticRegression(random_state=1)

# Fit the model using the rndovr training data
logreg_rndovr_model.fit(X_train_rndovr_df, y_train_rndovr_srs)

# Make a prediction using the testing data
logreg_y_pred_rndovr_npa = logreg_rndovr_model.predict(X_test_scaled_npa)


# ### Step 2a-b: Use the `SVM` classifier and the oversampled resampled data to fit the model and make predictions.

# In[ ]:


#### # Instantiate the model
#### svm_rndovr_model = SVC(kernel='linear')
#### 
#### # Fit the model using the rndovr training data
#### svm_rndovr_model.fit(X_train_rndovr_df, y_train_rndovr_srs)
#### 
#### # Make a prediction using the testing data
#### svm_y_pred_rndovr_npa = svm_rndovr_model.predict(X_test_scaled_npa)


# ### Step 2a-c: Use the `Decision Tree` classifier and the oversampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Creating the decision tree classifier instance
dectree_rndovr_model = tree.DecisionTreeClassifier(random_state=1)

# Fit the model using the rndovr training data
dectree_rndovr_model.fit(X_train_rndovr_df, y_train_rndovr_srs)

# Make a prediction using the testing data
dectree_y_pred_rndovr_npa = dectree_rndovr_model.predict(X_test_scaled_npa)


# ### Step 2a-d: Use the `Random Forest` classifier and the oversampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Create a random forest classifier
rndfor_rndovr_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fit the model using the rndovr training data
rndfor_rndovr_model.fit(X_train_rndovr_df, y_train_rndovr_srs)

# Make a prediction using the testing data
rndfor_y_pred_rndovr_npa = rndfor_rndovr_model.predict(X_test_scaled_npa)


# ### Step 2a-e: Use the `KNN` classifier and the oversampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Instantiate the model with k = 3 neighbors
knn_rndovr_model = KNeighborsClassifier(n_neighbors=3)

# Fit the model using the rndovr training data
knn_rndovr_model.fit(X_train_rndovr_df, y_train_rndovr_srs)

# Make a prediction using the testing data
knn_y_pred_rndovr_npa = knn_rndovr_model.predict(X_test_scaled_npa)


# ### Step 3a: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores (Random Oversampled) (w/ Scaling):")
print("-----------------------------------------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_rndovr_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_rndovr_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_rndovr_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_rndovr_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_rndovr_npa))


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_rndovr_npa, "Logistic Regression w/ Rand Oversampling (w/ Scaling)")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_rndovr_npa, "SVM w/ Rand Oversampling (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_rndovr_npa, "Decision Tree w/ Rand Oversampling (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_rndovr_npa, "Random Forest w/ Rand Oversampling (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_rndovr_npa, "KNN w/ Rand Oversampling (w/ Scaling)")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_rndovr_class_rpt = classification_report(y_test_srs, logreg_y_pred_rndovr_npa)

#### # Print the classification report for the SVM model
#### svm_rndovr_class_rpt = classification_report(y_test_srs, svm_y_pred_rndovr_npa)

# Print the classification report for the Decision Tree model
dectree_rndovr_class_rpt = classification_report(y_test_srs, dectree_y_pred_rndovr_npa)

# Print the classification report for the Random Forest model
rndfor_rndovr_class_rpt = classification_report(y_test_srs, rndfor_y_pred_rndovr_npa)

# Print the classification report for the KNN model
knn_rndovr_class_rpt = classification_report(y_test_srs, knn_y_pred_rndovr_npa)


# In[ ]:


print("Logistic Regression (w/ Random Oversampling) Classification Report (w/ Scaling):\n", logreg_rndovr_class_rpt)
####print("SVM (ww/ Random Oversampling) Classification Report (w/ Scaling):\n", svm_rndovr_class_rpt)
print("Decision Tree (ww/ Random Oversampling) Classification Report (w/ Scaling):\n", dectree_rndovr_class_rpt)
print("Random Forest (ww/ Random Oversampling) Classification Report (w/ Scaling):\n", rndfor_rndovr_class_rpt)
print("KNN (ww/ Random Oversampling) Classification Report (w/ Scaling):\n", knn_rndovr_class_rpt)


# ### Step 1b: Use the `RandomUnderSampler` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

# In[ ]:


# Instantiate the random undersampler model
random_undersampler = RandomUnderSampler(random_state=1)

# Fit the original training data to the random_undersampler model
X_train_rndunder_df, y_train_rndunder_srs = random_undersampler.fit_resample(X_train_scaled_npa, y_train_srs)


# In[ ]:


# Count the distinct values of the resampled labels data
print("# of random under sampled loan_status values that =0 (loan approved) and =1 (loan rejected)")
display(y_train_rndunder_srs.value_counts())


# ### Step 2b-a: Use the `LogisticRegression` classifier and the undersampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
logreg_rndunder_model = LogisticRegression(random_state=1)

# Fit the model using the rndunder training data
logreg_rndunder_model.fit(X_train_rndunder_df, y_train_rndunder_srs)

# Make a prediction using the testing data
logreg_y_pred_rndunder_npa = logreg_rndunder_model.predict(X_test_scaled_npa)


# ### Step 2b-b: Use the `SVM` classifier and the undersampled resampled data to fit the model and make predictions.

# In[ ]:


#### # Instantiate the model
#### svm_rndunder_model = SVC(kernel='linear')
#### 
#### # Fit the model using the rndunder training data
#### svm_rndunder_model.fit(X_train_rndunder_df, y_train_rndunder_srs)
#### 
#### # Make a prediction using the testing data
#### svm_y_pred_rndunder_npa = svm_rndunder_model.predict(X_test_scaled_npa)


# ### Step 2b-c: Use the `Decision Tree` classifier and the undersampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Creating the decision tree classifier instance
dectree_rndunder_model = tree.DecisionTreeClassifier(random_state=1)

# Fit the model using the rndunder training data
dectree_rndunder_model.fit(X_train_rndunder_df, y_train_rndunder_srs)

# Make a prediction using the testing data
dectree_y_pred_rndunder_npa = dectree_rndunder_model.predict(X_test_scaled_npa)


# ### Step 2b-d: Use the `Random Forest` classifier and the undersampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Create a random forest classifier
rndfor_rndunder_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fit the model using the rndunder training data
rndfor_rndunder_model.fit(X_train_rndunder_df, y_train_rndunder_srs)

# Make a prediction using the testing data
rndfor_y_pred_rndunder_npa = rndfor_rndunder_model.predict(X_test_scaled_npa)


# ### Step 2b-e: Use the `KNN` classifier and the undersampled resampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Instantiate the model with k = 3 neighbors
knn_rndunder_model = KNeighborsClassifier(n_neighbors=3)

# Fit the model using the rndunder training data
knn_rndunder_model.fit(X_train_rndunder_df, y_train_rndunder_srs)

# Make a prediction using the testing data
knn_y_pred_rndunder_npa = knn_rndunder_model.predict(X_test_scaled_npa)


# ### Step 3b: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores (Random Undersampled) (w/ Scaling):")
print("------------------------------------------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_rndunder_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_rndunder_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_rndunder_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_rndunder_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_rndunder_npa))


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_rndunder_npa, "Logistic Regression w/ Rand Undersampling (w/ Scaling)")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_rndunder_npa, "SVM w/ Rand Undersampling (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_rndunder_npa, "Decision Tree w/ Rand Undersampling (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_rndunder_npa, "Random Forest w/ Rand Undersampling (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_rndunder_npa, "KNN w/ Rand Undersampling (w/ Scaling)")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_rndunder_class_rpt = classification_report(y_test_srs, logreg_y_pred_rndunder_npa)

#### # Print the classification report for the SVM model
#### svm_rndunder_class_rpt = classification_report(y_test_srs, svm_y_pred_rndunder_npa)

# Print the classification report for the Decision Tree model
dectree_rndunder_class_rpt = classification_report(y_test_srs, dectree_y_pred_rndunder_npa)

# Print the classification report for the Random Forest model
rndfor_rndunder_class_rpt = classification_report(y_test_srs, rndfor_y_pred_rndunder_npa)

# Print the classification report for the KNN model
knn_rndunder_class_rpt = classification_report(y_test_srs, knn_y_pred_rndunder_npa)


# In[ ]:


print("Logistic Regression (w/ Random Undersampling) Classification Report (w/ Scaling):\n", logreg_rndunder_class_rpt)
####print("SVM (ww/ Random Undersampling) Classification Report (w/ Scaling):\n", svm_rndunder_class_rpt)
print("Decision Tree (ww/ Random Undersampling) Classification Report (w/ Scaling):\n", dectree_rndunder_class_rpt)
print("Random Forest (ww/ Random Undersampling) Classification Report (w/ Scaling):\n", rndfor_rndunder_class_rpt)
print("KNN (ww/ Random Undersampling) Classification Report (w/ Scaling):\n", knn_rndunder_class_rpt)


# ### Step 1c: Use the `SMOTE Oversampling` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

# In[ ]:


# Instantiate a SMOTE instance
smote_sampler =  SMOTE(random_state=1, sampling_strategy='auto')
# Fit the training data to the SMOTE model
X_train_smote_df, y_train_smote_srs = smote_sampler.fit_resample(X_train_scaled_npa, y_train_srs)
# Count distinct values for the resampled target data


# In[ ]:


# Count the distinct values of the resampled labels data
print("# of SMOTE Sampled loan_status values that =0 (loan approved) and =1 (loan rejected)")
display(y_train_smote_srs.value_counts())


# ### Step 2c-a: Use the `LogisticRegression` classifier and the smote sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
logreg_smote_model = LogisticRegression(random_state=1)

# Fit the model using the smote training data
logreg_smote_model.fit(X_train_smote_df, y_train_smote_srs)

# Make a prediction using the testing data
logreg_y_pred_smote_npa = logreg_smote_model.predict(X_test_scaled_npa)


# ### Step 2c-b: Use the `SVM` classifier and the smote sampled data to fit the model and make predictions.

# In[ ]:


#### # Instantiate the model
#### svm_smote_model = SVC(kernel='linear')
#### 
#### # Fit the model using the smote training data
#### svm_smote_model.fit(X_train_smote_df, y_train_smote_srs)
#### 
#### # Make a prediction using the testing data
#### svm_y_pred_smote_npa = svm_smote_model.predict(X_test_scaled_npa)


# ### Step 2c-c: Use the `Decision Tree` classifier and the smote sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Creating the decision tree classifier instance
dectree_smote_model = tree.DecisionTreeClassifier(random_state=1)

# Fit the model using the smote training data
dectree_smote_model.fit(X_train_smote_df, y_train_smote_srs)

# Make a prediction using the testing data
dectree_y_pred_smote_npa = dectree_smote_model.predict(X_test_scaled_npa)


# ### Step 2c-d: Use the `Random Forest` classifier and the smote sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Create a random forest classifier
rndfor_smote_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fit the model using the smote training data
rndfor_smote_model.fit(X_train_smote_df, y_train_smote_srs)

# Make a prediction using the testing data
rndfor_y_pred_smote_npa = rndfor_smote_model.predict(X_test_scaled_npa)


# ### Step 2c-e: Use the `KNN` classifier and the smote sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Instantiate the model with k = 3 neighbors
knn_smote_model = KNeighborsClassifier(n_neighbors=3)

# Fit the model using the smote training data
knn_smote_model.fit(X_train_smote_df, y_train_smote_srs)

# Make a prediction using the testing data
knn_y_pred_smote_npa = knn_smote_model.predict(X_test_scaled_npa)


# ### Step 3c: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores (SMOTE Sampled) (w/ Scaling):")
print("------------------------------------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_smote_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_smote_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_smote_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_smote_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_smote_npa))


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_smote_npa, "Logistic Regression w/ SMOTE Sampled (w/ Scaling)")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_smote_npa, "SVM w/ SMOTE Sampled (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_smote_npa, "Decision Tree w/ SMOTE Sampled (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_smote_npa, "Random Forest w/ SMOTE Sampled (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_smote_npa, "KNN w/ SMOTE Sampled (w/ Scaling)")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_smote_class_rpt = classification_report(y_test_srs, logreg_y_pred_smote_npa)

#### # Print the classification report for the SVM model
#### svm_smote_class_rpt = classification_report(y_test_srs, svm_y_pred_smote_npa)

# Print the classification report for the Decision Tree model
dectree_smote_class_rpt = classification_report(y_test_srs, dectree_y_pred_smote_npa)

# Print the classification report for the Random Forest model
rndfor_smote_class_rpt = classification_report(y_test_srs, rndfor_y_pred_smote_npa)

# Print the classification report for the KNN model
knn_smote_class_rpt = classification_report(y_test_srs, knn_y_pred_smote_npa)


# In[ ]:


print("Logistic Regression (w/ SMOTE Sampled) Classification Report (w/ Scaling):\n", logreg_smote_class_rpt)
####print("SVM (ww/ SMOTE Sampled) Classification Report (w/ Scaling):\n", svm_smote_class_rpt)
print("Decision Tree (ww/ SMOTE Sampled)g Classification Report (w/ Scaling):\n", dectree_smote_class_rpt)
print("Random Forest (ww/ SMOTE Sampled)g Classification Report (w/ Scaling):\n", rndfor_smote_class_rpt)
print("KNN (ww/ SMOTE Sampled) Classification Report (w/ Scaling):\n", knn_smote_class_rpt)


# ### Step 1d: Use the `SMOTEENN Combined Sampling` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

# In[ ]:


# Instantiate a SMOTEENN instance
smoteenn_sampler =  SMOTEENN(random_state=1)
# Fit the training data to the SMOTEENN model
X_train_smoteenn_df, y_train_smoteenn_srs = smoteenn_sampler.fit_resample(X_train_scaled_npa, y_train_srs)
# Count distinct values for the resampled target data


# In[ ]:


# Count the distinct values of the resampled labels data
print("# of SMOTEENN Sampled loan_status values that =0 (loan approved) and =1 (loan rejected)")
display(y_train_smoteenn_srs.value_counts())


# ### Step 2d-a: Use the `LogisticRegression` classifier and the smoteenn sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
logreg_smoteenn_model = LogisticRegression(random_state=1)

# Fit the model using the smoteenn training data
logreg_smoteenn_model.fit(X_train_smoteenn_df, y_train_smoteenn_srs)

# Make a prediction using the testing data
logreg_y_pred_smoteenn_npa = logreg_smoteenn_model.predict(X_test_scaled_npa)


# ### Step 2d-b: Use the `SVM` classifier and the smoteenn sampled data to fit the model and make predictions.

# In[ ]:


#### # Instantiate the model
#### svm_smoteenn_model = SVC(kernel='linear')
#### 
#### # Fit the model using the smoteenn training data
#### svm_smoteenn_model.fit(X_train_smoteenn_df, y_train_smoteenn_srs)
#### 
#### # Make a prediction using the testing data
#### svm_y_pred_smoteenn_npa = svm_smoteenn_model.predict(X_tesX_test_scaled_npat_df)


# ### Step 2d-c: Use the `Decision Tree` classifier and the smoteenn sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Creating the decision tree classifier instance
dectree_smoteenn_model = tree.DecisionTreeClassifier(random_state=1)

# Fit the model using the smoteenn training data
dectree_smoteenn_model.fit(X_train_smoteenn_df, y_train_smoteenn_srs)

# Make a prediction using the testing data
dectree_y_pred_smoteenn_npa = dectree_smoteenn_model.predict(X_test_scaled_npa)


# ### Step 2d-d: Use the `Random Forest` classifier and the smoteenn sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Create a random forest classifier
rndfor_smoteenn_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fit the model using the smoteenn training data
rndfor_smoteenn_model.fit(X_train_smoteenn_df, y_train_smoteenn_srs)

# Make a prediction using the testing data
rndfor_y_pred_smoteenn_npa = rndfor_smoteenn_model.predict(X_test_scaled_npa)


# ### Step 2d-e: Use the `KNN` classifier and the smoteenn sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Instantiate the model with k = 3 neighbors
knn_smoteenn_model = KNeighborsClassifier(n_neighbors=3)

# Fit the model using the smoteenn training data
knn_smoteenn_model.fit(X_train_smoteenn_df, y_train_smoteenn_srs)

# Make a prediction using the testing data
knn_y_pred_smoteenn_npa = knn_smoteenn_model.predict(X_test_scaled_npa)


# ### Step 3d: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores (SMOTEENN Sampled) (w/ Scaling):")
print("---------------------------------------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_smoteenn_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_smoteenn_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_smoteenn_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_smoteenn_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_smoteenn_npa))


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_smoteenn_npa, "Logistic Regression w/ SMOTEENN Sampled (w/ Scaling)")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_smoteenn_npa, "SVM w/ SMOTEENN Sampled (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_smoteenn_npa, "Decision Tree w/ SMOTEENN Sampled (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_smoteenn_npa, "Random Forest w/ SMOTEENN Sampled (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_smoteenn_npa, "KNN w/ SMOTEENN Sampled (w/ Scaling)")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_smoteenn_class_rpt = classification_report(y_test_srs, logreg_y_pred_smoteenn_npa)

#### # Print the classification report for the SVM model
#### svm_smoteenn_class_rpt = classification_report(y_test_srs, svm_y_pred_smoteenn_npa)

# Print the classification report for the Decision Tree model
dectree_smoteenn_class_rpt = classification_report(y_test_srs, dectree_y_pred_smoteenn_npa)

# Print the classification report for the Random Forest model
rndfor_smoteenn_class_rpt = classification_report(y_test_srs, rndfor_y_pred_smoteenn_npa)

# Print the classification report for the KNN model
knn_smoteenn_class_rpt = classification_report(y_test_srs, knn_y_pred_smoteenn_npa)


# In[ ]:


print("Logistic Regression (w/ SMOTEENN Sampled) Classification Report (w/ Scaling):\n", logreg_smoteenn_class_rpt)
####print("SVM (ww/ SMOTEENN Sampled) Classification Report (w/ Scaling):\n", svm_smoteenn_class_rpt)
print("Decision Tree (ww/ SMOTEENN Sampled)g Classification Report (w/ Scaling):\n", dectree_smoteenn_class_rpt)
print("Random Forest (ww/ SMOTEENN Sampled)g Classification Report (w/ Scaling):\n", rndfor_smoteenn_class_rpt)
print("KNN (ww/ SMOTEENN Sampled) Classification Report (w/ Scaling):\n", knn_smoteenn_class_rpt)


# ### Step e: Use the `ClusterCentroids` module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points. 

# In[ ]:


# Instantiate a ClusterCentroids instance
cluster_centroid_sampler = ClusterCentroids(random_state=1)
# Fit the training data to the cluster centroids model
X_train_clstrcntrd_df, y_train_clstrcntrd_srs = cluster_centroid_sampler.fit_resample(X_train_scaled_npa, y_train_srs)
# Count distinct values for the resampled target data


# In[ ]:


# Count the distinct values of the resampled labels data
print("# of cluster centroid loan_status values that =0 (loan approved) and =1 (loan rejected)")
display(y_train_clstrcntrd_srs.value_counts())


# ### Step 2e-a: Use the `LogisticRegression` classifier and the cluster centroid sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
logreg_clstrcntrd_model = LogisticRegression(random_state=1)

# Fit the model using the clstrcntrd training data
logreg_clstrcntrd_model.fit(X_train_clstrcntrd_df, y_train_clstrcntrd_srs)

# Make a prediction using the testing data
logreg_y_pred_clstrcntrd_npa = logreg_clstrcntrd_model.predict(X_test_scaled_npa)


# ### Step 2e-b: Use the `SVM` classifier and the cluster centroid sampled data to fit the model and make predictions.

# In[ ]:


#### # Instantiate the model
#### svm_clstrcntrd_model = SVC(kernel='linear')
#### 
#### # Fit the model using the clstrcntrd training data
#### svm_clstrcntrd_model.fit(X_train_clstrcntrd_df, y_train_clstrcntrd_srs)
#### 
#### # Make a prediction using the testing data
#### svm_y_pred_clstrcntrd_npa = svm_clstrcntrd_model.predict(X_test_scaled_npa)


# ### Step 2e-c: Use the `Decision Tree` classifier and the cluster centroid sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Creating the decision tree classifier instance
dectree_clstrcntrd_model = tree.DecisionTreeClassifier(random_state=1)

# Fit the model using the clstrcntrd training data
dectree_clstrcntrd_model.fit(X_train_clstrcntrd_df, y_train_clstrcntrd_srs)

# Make a prediction using the testing data
dectree_y_pred_clstrcntrd_npa = dectree_clstrcntrd_model.predict(X_test_scaled_npa)


# ### Step 2e-d: Use the `Random Forest` classifier and the cluster centroid sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Create a random forest classifier
rndfor_clstrcntrd_model = RandomForestClassifier(n_estimators=100, random_state=1)

# Fit the model using the clstrcntrd training data
rndfor_clstrcntrd_model.fit(X_train_clstrcntrd_df, y_train_clstrcntrd_srs)

# Make a prediction using the testing data
rndfor_y_pred_clstrcntrd_npa = rndfor_clstrcntrd_model.predict(X_test_scaled_npa)


# ### Step 2e-e: Use the `KNN` classifier and the cluster centroid sampled data to fit the model and make predictions.

# In[ ]:


# Instantiate the model
# Instantiate the model with k = 3 neighbors
knn_clstrcntrd_model = KNeighborsClassifier(n_neighbors=3)

# Fit the model using the clstrcntrd training data
knn_clstrcntrd_model.fit(X_train_clstrcntrd_df, y_train_clstrcntrd_srs)

# Make a prediction using the testing data
knn_y_pred_clstrcntrd_npa = knn_clstrcntrd_model.predict(X_test_scaled_npa)


# ### Step 3e: Evaluate the model’s performance by doing the following:
# 
# * Calculate the accuracy score of the model.
# 
# * Generate a confusion matrix.
# 
# * Print the classification report.

# In[ ]:


print("Balanced Accuracy Scores (Cluster Centroid Sampled) (w/ Scaling):")
print("-----------------------------------------------------------------")

# Print the balanced_accuracy score of the Logistic Regression Model
print("Logistic Regression:                  ", balanced_accuracy_score(y_test_srs, logreg_y_pred_clstrcntrd_npa))

#### # Print the balanced_accuracy score of the Support Vector Matrix Model
#### print("Support Vector Machine:               ", balanced_accuracy_score(y_test_srs, svm_y_pred_clstrcntrd_npa))

# Print the balanced_accuracy score of the Decision Tree Model
print("Decision Tree:                        ", balanced_accuracy_score(y_test_srs, dectree_y_pred_clstrcntrd_npa))

# Print the balanced_accuracy score of the Random Forest Model
print("Random Forest:                        ", balanced_accuracy_score(y_test_srs, rndfor_y_pred_clstrcntrd_npa))

# Print the balanced_accuracy score of the KNN Model
print("KNN:                                  ", balanced_accuracy_score(y_test_srs, knn_y_pred_clstrcntrd_npa))


# In[ ]:


confusion_matrix_sklearn(y_test_srs, logreg_y_pred_clstrcntrd_npa, "Logistic Regression w/ Cluster Centroid Sampled (w/ Scaling)")
#### confusion_matrix_sklearn(y_test_srs, svm_y_pred_clstrcntrd_npa, "SVM w/ Cluster Centroid Sampled (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, dectree_y_pred_clstrcntrd_npa, "Decision Tree w/ Cluster Centroid Sampled (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, rndfor_y_pred_clstrcntrd_npa, "Random Forest w/ Cluster Centroid Sampled (w/ Scaling)")
confusion_matrix_sklearn(y_test_srs, knn_y_pred_clstrcntrd_npa, "KNN w/ Cluster Centroid Sampled (w/ Scaling)")


# In[ ]:


# Print the classification report for the Logistics Regression model
logreg_clstrcntrd_class_rpt = classification_report(y_test_srs, logreg_y_pred_clstrcntrd_npa)

#### # Print the classification report for the SVM model
#### svm_clstrcntrd_class_rpt = classification_report(y_test_srs, svm_y_pred_clstrcntrd_npa)

# Print the classification report for the Decision Tree model
dectree_clstrcntrd_class_rpt = classification_report(y_test_srs, dectree_y_pred_clstrcntrd_npa)

# Print the classification report for the Random Forest model
rndfor_clstrcntrd_class_rpt = classification_report(y_test_srs, rndfor_y_pred_clstrcntrd_npa)

# Print the classification report for the KNN model
knn_clstrcntrd_class_rpt = classification_report(y_test_srs, knn_y_pred_clstrcntrd_npa)


# In[ ]:


print("Logistic Regression (w/ Cluster Centroid Sampled) Classification Repor (w/ Scaling)t:\n", logreg_clstrcntrd_class_rpt)
####print("SVM (ww/ Cluster Centroid Sampled) Classification Report (w/ Scaling):\n", svm_clstrcntrd_class_rpt)
print("Decision Tree (ww/ Cluster Centroid Sampled)g Classification Report (w/ Scaling):\n", dectree_clstrcntrd_class_rpt)
print("Random Forest (ww/ Cluster Centroid Sampled)g Classification Report (w/ Scaling):\n", rndfor_clstrcntrd_class_rpt)
print("KNN (ww/ Cluster Centroid Sampled) Classification Report (w/ Scaling):\n", knn_clstrcntrd_class_rpt)


# In[ ]:


# (Re-)Calculate classificaiton report to output to a dictionary format

# Logistic Regression
logreg_class_rpt_scaled_dict = classification_report(y_test_srs, logreg_y_pred_npa, output_dict = True)
#### svm_class_rpt_scaled_dict = classification_report(y_test_srs, svm_y_pred_npa, output_dict = True)
dectree_class_rpt_scaled_dict = classification_report(y_test_srs, dectree_y_pred_npa, output_dict = True)
rndfor_class_rpt_scaled_dict = classification_report(y_test_srs, rndfor_y_pred_npa, output_dict = True)
knn_class_rpt_scaled_dict = classification_report(y_test_srs, knn_y_pred_npa, output_dict = True)

logreg_rndovr_class_rpt_scaled_dict = classification_report(y_test_srs, logreg_y_pred_rndovr_npa, output_dict = True)
#### svm_rndovr_class_rpt_scaled_dict = classification_report(y_test_srs, svm_y_pred_rndovr_np, output_dict = Truea)
dectree_rndovr_class_rpt_scaled_dict = classification_report(y_test_srs, dectree_y_pred_rndovr_npa, output_dict = True)
rndfor_rndovr_class_rpt_scaled_dict = classification_report(y_test_srs, rndfor_y_pred_rndovr_npa, output_dict = True)
knn_rndovr_class_rpt_scaled_dict = classification_report(y_test_srs, knn_y_pred_rndovr_npa, output_dict = True)

logreg_rndunder_class_rpt_scaled_dict = classification_report(y_test_srs, logreg_y_pred_rndunder_npa, output_dict = True)
#### svm_rndunder_class_rpt_scaled_dict = classification_report(y_test_srs, svm_y_pred_rndunder_npa, output_dict = True)
dectree_rndunder_class_rpt_scaled_dict = classification_report(y_test_srs, dectree_y_pred_rndunder_npa, output_dict = True)
rndfor_rndunder_class_rpt_scaled_dict = classification_report(y_test_srs, rndfor_y_pred_rndunder_npa, output_dict = True)
knn_rndunder_class_rpt_scaled_dict = classification_report(y_test_srs, knn_y_pred_rndunder_npa, output_dict = True)

logreg_smote_class_rpt_scaled_dict = classification_report(y_test_srs, logreg_y_pred_smote_npa, output_dict = True)
#### svm_smote_class_rpt_scaled_dict = classification_report(y_test_srs, svm_y_pred_smote_npa, output_dict = True)
dectree_smote_class_rpt_scaled_dict = classification_report(y_test_srs, dectree_y_pred_smote_npa, output_dict = True)
rndfor_smote_class_rpt_scaled_dict = classification_report(y_test_srs, rndfor_y_pred_smote_npa, output_dict = True)
knn_smote_class_rpt_scaled_dict = classification_report(y_test_srs, knn_y_pred_smote_npa, output_dict = True)

logreg_smoteenn_class_rpt_scaled_dict = classification_report(y_test_srs, logreg_y_pred_smoteenn_npa, output_dict = True)
#### svm_smoteenn_class_rpt_scaled_dict = classification_report(y_test_srs, svm_y_pred_smoteenn_npa, output_dict = True)
dectree_smoteenn_class_rpt_scaled_dict = classification_report(y_test_srs, dectree_y_pred_smoteenn_npa, output_dict = True)
rndfor_smoteenn_class_rpt_scaled_dict = classification_report(y_test_srs, rndfor_y_pred_smoteenn_npa, output_dict = True)
knn_smoteenn_class_rpt_scaled_dict = classification_report(y_test_srs, knn_y_pred_smoteenn_npa, output_dict = True)

logreg_clstrcntrd_class_rpt_scaled_dict = classification_report(y_test_srs, logreg_y_pred_clstrcntrd_npa, output_dict = True)
#### svm_clstrcntrd_class_rpt_scaled_dict = classification_report(y_test_srs, svm_y_pred_clstrcntrd_npa, output_dict = True)
dectree_clstrcntrd_class_rpt_scaled_dict = classification_report(y_test_srs, dectree_y_pred_clstrcntrd_npa, output_dict = True)
rndfor_clstrcntrd_class_rpt_scaled_dict = classification_report(y_test_srs, rndfor_y_pred_clstrcntrd_npa, output_dict = True)
knn_clstrcntrd_class_rpt_scaled_dict = classification_report(y_test_srs, knn_y_pred_clstrcntrd_npa, output_dict = True)


# In[ ]:





# In[ ]:


accuracy_npa = np.array([logreg_class_rpt_dict['accuracy'],dectree_class_rpt_dict['accuracy'],rndfor_class_rpt_dict['accuracy'],knn_class_rpt_dict['accuracy'],
                         logreg_rndovr_class_rpt_dict['accuracy'],dectree_rndovr_class_rpt_dict['accuracy'],rndfor_rndovr_class_rpt_dict['accuracy'],knn_rndovr_class_rpt_dict['accuracy'],
                         logreg_rndunder_class_rpt_dict['accuracy'],dectree_rndunder_class_rpt_dict['accuracy'],rndfor_rndunder_class_rpt_dict['accuracy'],knn_rndunder_class_rpt_dict['accuracy'],
                         logreg_smote_class_rpt_dict['accuracy'],dectree_smote_class_rpt_dict['accuracy'],rndfor_smote_class_rpt_dict['accuracy'],knn_smote_class_rpt_dict['accuracy'],
                         logreg_smoteenn_class_rpt_dict['accuracy'],dectree_smoteenn_class_rpt_dict['accuracy'],rndfor_smoteenn_class_rpt_dict['accuracy'],knn_smoteenn_class_rpt_dict['accuracy'],
                         logreg_clstrcntrd_class_rpt_dict['accuracy'],dectree_clstrcntrd_class_rpt_dict['accuracy'],rndfor_clstrcntrd_class_rpt_dict['accuracy'],knn_clstrcntrd_class_rpt_dict['accuracy']
                        ])

precision_0_npa = np.array([logreg_class_rpt_dict['0']['precision'],dectree_class_rpt_dict['0']['precision'],rndfor_class_rpt_dict['0']['precision'],knn_class_rpt_dict['0']['precision'],
                            logreg_rndovr_class_rpt_dict['0']['precision'],dectree_rndovr_class_rpt_dict['0']['precision'],rndfor_rndovr_class_rpt_dict['0']['precision'],knn_rndovr_class_rpt_dict['0']['precision'],
                            logreg_rndunder_class_rpt_dict['0']['precision'],dectree_rndunder_class_rpt_dict['0']['precision'],rndfor_rndunder_class_rpt_dict['0']['precision'],knn_rndunder_class_rpt_dict['0']['precision'],
                            logreg_smote_class_rpt_dict['0']['precision'],dectree_smote_class_rpt_dict['0']['precision'],rndfor_smote_class_rpt_dict['0']['precision'],knn_smote_class_rpt_dict['0']['precision'],
                            logreg_smoteenn_class_rpt_dict['0']['precision'],dectree_smoteenn_class_rpt_dict['0']['precision'],rndfor_smoteenn_class_rpt_dict['0']['precision'],knn_smoteenn_class_rpt_dict['0']['precision'],
                            logreg_clstrcntrd_class_rpt_dict['0']['precision'],dectree_clstrcntrd_class_rpt_dict['0']['precision'],rndfor_clstrcntrd_class_rpt_dict['0']['precision'],knn_clstrcntrd_class_rpt_dict['0']['precision']
                           ])
precision_1_npa = np.array([logreg_class_rpt_dict['1']['precision'],dectree_class_rpt_dict['1']['precision'],rndfor_class_rpt_dict['1']['precision'],knn_class_rpt_dict['1']['precision'],
                            logreg_rndovr_class_rpt_dict['1']['precision'],dectree_rndovr_class_rpt_dict['1']['precision'],rndfor_rndovr_class_rpt_dict['1']['precision'],knn_rndovr_class_rpt_dict['1']['precision'],
                            logreg_rndunder_class_rpt_dict['1']['precision'],dectree_rndunder_class_rpt_dict['1']['precision'],rndfor_rndunder_class_rpt_dict['1']['precision'],knn_rndunder_class_rpt_dict['1']['precision'],
                            logreg_smote_class_rpt_dict['1']['precision'],dectree_smote_class_rpt_dict['1']['precision'],rndfor_smote_class_rpt_dict['1']['precision'],knn_smote_class_rpt_dict['1']['precision'],
                            logreg_smoteenn_class_rpt_dict['1']['precision'],dectree_smoteenn_class_rpt_dict['1']['precision'],rndfor_smoteenn_class_rpt_dict['1']['precision'],knn_smoteenn_class_rpt_dict['1']['precision'],
                            logreg_clstrcntrd_class_rpt_dict['1']['precision'],dectree_clstrcntrd_class_rpt_dict['1']['precision'],rndfor_clstrcntrd_class_rpt_dict['1']['precision'],knn_clstrcntrd_class_rpt_dict['1']['precision']
                           ])

recall_0_npa = np.array([logreg_class_rpt_dict['0']['recall'],dectree_class_rpt_dict['0']['recall'],rndfor_class_rpt_dict['0']['recall'],knn_class_rpt_dict['0']['recall'],
                         logreg_rndovr_class_rpt_dict['0']['recall'],dectree_rndovr_class_rpt_dict['0']['recall'],rndfor_rndovr_class_rpt_dict['0']['recall'],knn_rndovr_class_rpt_dict['0']['recall'],
                         logreg_rndunder_class_rpt_dict['0']['recall'],dectree_rndunder_class_rpt_dict['0']['recall'],rndfor_rndunder_class_rpt_dict['0']['recall'],knn_rndunder_class_rpt_dict['0']['recall'],
                         logreg_smote_class_rpt_dict['0']['recall'],dectree_smote_class_rpt_dict['0']['recall'],rndfor_smote_class_rpt_dict['0']['recall'],knn_smote_class_rpt_dict['0']['recall'],
                         logreg_smoteenn_class_rpt_dict['0']['recall'],dectree_smoteenn_class_rpt_dict['0']['recall'],rndfor_smoteenn_class_rpt_dict['0']['recall'],knn_smoteenn_class_rpt_dict['0']['recall'],
                         logreg_clstrcntrd_class_rpt_dict['0']['recall'],dectree_clstrcntrd_class_rpt_dict['0']['recall'],rndfor_clstrcntrd_class_rpt_dict['0']['recall'],knn_clstrcntrd_class_rpt_dict['0']['recall']
                           ])
recall_1_npa = np.array([logreg_class_rpt_dict['1']['recall'],dectree_class_rpt_dict['1']['recall'],rndfor_class_rpt_dict['1']['recall'],knn_class_rpt_dict['1']['recall'],
                         logreg_rndovr_class_rpt_dict['1']['recall'],dectree_rndovr_class_rpt_dict['1']['recall'],rndfor_rndovr_class_rpt_dict['1']['recall'],knn_rndovr_class_rpt_dict['1']['recall'],
                         logreg_rndunder_class_rpt_dict['1']['recall'],dectree_rndunder_class_rpt_dict['1']['recall'],rndfor_rndunder_class_rpt_dict['1']['recall'],knn_rndunder_class_rpt_dict['1']['recall'],
                         logreg_smote_class_rpt_dict['1']['recall'],dectree_smote_class_rpt_dict['1']['recall'],rndfor_smote_class_rpt_dict['1']['recall'],knn_smote_class_rpt_dict['1']['recall'],
                         logreg_smoteenn_class_rpt_dict['1']['recall'],dectree_smoteenn_class_rpt_dict['1']['recall'],rndfor_smoteenn_class_rpt_dict['1']['recall'],knn_smoteenn_class_rpt_dict['1']['recall'],
                          logreg_clstrcntrd_class_rpt_dict['1']['recall'],dectree_clstrcntrd_class_rpt_dict['1']['recall'],rndfor_clstrcntrd_class_rpt_dict['1']['recall'],knn_clstrcntrd_class_rpt_dict['1']['recall']
                           ])

f1_score_0_npa = np.array([logreg_class_rpt_dict['0']['f1-score'],dectree_class_rpt_dict['0']['f1-score'],rndfor_class_rpt_dict['0']['f1-score'],knn_class_rpt_dict['0']['f1-score'],
                           logreg_rndovr_class_rpt_dict['0']['f1-score'],dectree_rndovr_class_rpt_dict['0']['f1-score'],rndfor_rndovr_class_rpt_dict['0']['f1-score'],knn_rndovr_class_rpt_dict['0']['f1-score'],
                           logreg_rndunder_class_rpt_dict['0']['f1-score'],dectree_rndunder_class_rpt_dict['0']['f1-score'],rndfor_rndunder_class_rpt_dict['0']['f1-score'],knn_rndunder_class_rpt_dict['0']['f1-score'],
                           logreg_smote_class_rpt_dict['0']['f1-score'],dectree_smote_class_rpt_dict['0']['f1-score'],rndfor_smote_class_rpt_dict['0']['f1-score'],knn_smote_class_rpt_dict['0']['f1-score'],
                           logreg_smoteenn_class_rpt_dict['0']['f1-score'],dectree_smoteenn_class_rpt_dict['0']['f1-score'],rndfor_smoteenn_class_rpt_dict['0']['f1-score'],knn_smoteenn_class_rpt_dict['0']['f1-score'],
                           logreg_clstrcntrd_class_rpt_dict['0']['f1-score'],dectree_clstrcntrd_class_rpt_dict['0']['f1-score'],rndfor_clstrcntrd_class_rpt_dict['0']['f1-score'],knn_clstrcntrd_class_rpt_dict['0']['f1-score']
                           ])
f1_score_1_npa = np.array([logreg_class_rpt_dict['1']['f1-score'],dectree_class_rpt_dict['1']['f1-score'],rndfor_class_rpt_dict['1']['f1-score'],knn_class_rpt_dict['1']['f1-score'],
                           logreg_rndovr_class_rpt_dict['1']['f1-score'],dectree_rndovr_class_rpt_dict['1']['f1-score'],rndfor_rndovr_class_rpt_dict['1']['f1-score'],knn_rndovr_class_rpt_dict['1']['f1-score'],
                           logreg_rndunder_class_rpt_dict['1']['f1-score'],dectree_rndunder_class_rpt_dict['1']['f1-score'],rndfor_rndunder_class_rpt_dict['1']['f1-score'],knn_rndunder_class_rpt_dict['1']['f1-score'],
                           logreg_smote_class_rpt_dict['1']['f1-score'],dectree_smote_class_rpt_dict['1']['f1-score'],rndfor_smote_class_rpt_dict['1']['f1-score'],knn_smote_class_rpt_dict['1']['f1-score'],
                           logreg_smoteenn_class_rpt_dict['1']['f1-score'],dectree_smoteenn_class_rpt_dict['1']['f1-score'],rndfor_smoteenn_class_rpt_dict['1']['f1-score'],knn_smoteenn_class_rpt_dict['1']['f1-score'],
                           logreg_clstrcntrd_class_rpt_dict['1']['f1-score'],dectree_clstrcntrd_class_rpt_dict['1']['f1-score'],rndfor_clstrcntrd_class_rpt_dict['1']['f1-score'],knn_clstrcntrd_class_rpt_dict['1']['f1-score']
                           ])



accuracy_scaled_npa = np.array([logreg_class_rpt_scaled_dict['accuracy'],dectree_class_rpt_scaled_dict['accuracy'],rndfor_class_rpt_scaled_dict['accuracy'],knn_class_rpt_scaled_dict['accuracy'],
                         logreg_rndovr_class_rpt_scaled_dict['accuracy'],dectree_rndovr_class_rpt_scaled_dict['accuracy'],rndfor_rndovr_class_rpt_scaled_dict['accuracy'],knn_rndovr_class_rpt_scaled_dict['accuracy'],
                         logreg_rndunder_class_rpt_scaled_dict['accuracy'],dectree_rndunder_class_rpt_scaled_dict['accuracy'],rndfor_rndunder_class_rpt_scaled_dict['accuracy'],knn_rndunder_class_rpt_scaled_dict['accuracy'],
                         logreg_smote_class_rpt_scaled_dict['accuracy'],dectree_smote_class_rpt_scaled_dict['accuracy'],rndfor_smote_class_rpt_scaled_dict['accuracy'],knn_smote_class_rpt_scaled_dict['accuracy'],
                         logreg_smoteenn_class_rpt_scaled_dict['accuracy'],dectree_smoteenn_class_rpt_scaled_dict['accuracy'],rndfor_smoteenn_class_rpt_scaled_dict['accuracy'],knn_smoteenn_class_rpt_scaled_dict['accuracy'],
                         logreg_clstrcntrd_class_rpt_scaled_dict['accuracy'],dectree_clstrcntrd_class_rpt_scaled_dict['accuracy'],rndfor_clstrcntrd_class_rpt_scaled_dict['accuracy'],knn_clstrcntrd_class_rpt_scaled_dict['accuracy']
                        ])

precision_0_scaled_npa = np.array([logreg_class_rpt_scaled_dict['0']['precision'],dectree_class_rpt_scaled_dict['0']['precision'],rndfor_class_rpt_scaled_dict['0']['precision'],knn_class_rpt_scaled_dict['0']['precision'],
                            logreg_rndovr_class_rpt_scaled_dict['0']['precision'],dectree_rndovr_class_rpt_scaled_dict['0']['precision'],rndfor_rndovr_class_rpt_scaled_dict['0']['precision'],knn_rndovr_class_rpt_scaled_dict['0']['precision'],
                            logreg_rndunder_class_rpt_scaled_dict['0']['precision'],dectree_rndunder_class_rpt_scaled_dict['0']['precision'],rndfor_rndunder_class_rpt_scaled_dict['0']['precision'],knn_rndunder_class_rpt_scaled_dict['0']['precision'],
                            logreg_smote_class_rpt_scaled_dict['0']['precision'],dectree_smote_class_rpt_scaled_dict['0']['precision'],rndfor_smote_class_rpt_scaled_dict['0']['precision'],knn_smote_class_rpt_scaled_dict['0']['precision'],
                            logreg_smoteenn_class_rpt_scaled_dict['0']['precision'],dectree_smoteenn_class_rpt_scaled_dict['0']['precision'],rndfor_smoteenn_class_rpt_scaled_dict['0']['precision'],knn_smoteenn_class_rpt_scaled_dict['0']['precision'],
                            logreg_clstrcntrd_class_rpt_scaled_dict['0']['precision'],dectree_clstrcntrd_class_rpt_scaled_dict['0']['precision'],rndfor_clstrcntrd_class_rpt_scaled_dict['0']['precision'],knn_clstrcntrd_class_rpt_scaled_dict['0']['precision']
                           ])
precision_1_scaled_npa = np.array([logreg_class_rpt_scaled_dict['1']['precision'],dectree_class_rpt_scaled_dict['1']['precision'],rndfor_class_rpt_scaled_dict['1']['precision'],knn_class_rpt_scaled_dict['1']['precision'],
                            logreg_rndovr_class_rpt_scaled_dict['1']['precision'],dectree_rndovr_class_rpt_scaled_dict['1']['precision'],rndfor_rndovr_class_rpt_scaled_dict['1']['precision'],knn_rndovr_class_rpt_scaled_dict['1']['precision'],
                            logreg_rndunder_class_rpt_scaled_dict['1']['precision'],dectree_rndunder_class_rpt_scaled_dict['1']['precision'],rndfor_rndunder_class_rpt_scaled_dict['1']['precision'],knn_rndunder_class_rpt_scaled_dict['1']['precision'],
                            logreg_smote_class_rpt_scaled_dict['1']['precision'],dectree_smote_class_rpt_scaled_dict['1']['precision'],rndfor_smote_class_rpt_scaled_dict['1']['precision'],knn_smote_class_rpt_scaled_dict['1']['precision'],
                            logreg_smoteenn_class_rpt_scaled_dict['1']['precision'],dectree_smoteenn_class_rpt_scaled_dict['1']['precision'],rndfor_smoteenn_class_rpt_scaled_dict['1']['precision'],knn_smoteenn_class_rpt_scaled_dict['1']['precision'],
                            logreg_clstrcntrd_class_rpt_scaled_dict['1']['precision'],dectree_clstrcntrd_class_rpt_scaled_dict['1']['precision'],rndfor_clstrcntrd_class_rpt_scaled_dict['1']['precision'],knn_clstrcntrd_class_rpt_scaled_dict['1']['precision']
                           ])

recall_0_scaled_npa = np.array([logreg_class_rpt_scaled_dict['0']['recall'],dectree_class_rpt_scaled_dict['0']['recall'],rndfor_class_rpt_scaled_dict['0']['recall'],knn_class_rpt_scaled_dict['0']['recall'],
                         logreg_rndovr_class_rpt_scaled_dict['0']['recall'],dectree_rndovr_class_rpt_scaled_dict['0']['recall'],rndfor_rndovr_class_rpt_scaled_dict['0']['recall'],knn_rndovr_class_rpt_scaled_dict['0']['recall'],
                         logreg_rndunder_class_rpt_scaled_dict['0']['recall'],dectree_rndunder_class_rpt_scaled_dict['0']['recall'],rndfor_rndunder_class_rpt_scaled_dict['0']['recall'],knn_rndunder_class_rpt_scaled_dict['0']['recall'],
                         logreg_smote_class_rpt_scaled_dict['0']['recall'],dectree_smote_class_rpt_scaled_dict['0']['recall'],rndfor_smote_class_rpt_scaled_dict['0']['recall'],knn_smote_class_rpt_scaled_dict['0']['recall'],
                         logreg_smoteenn_class_rpt_scaled_dict['0']['recall'],dectree_smoteenn_class_rpt_scaled_dict['0']['recall'],rndfor_smoteenn_class_rpt_scaled_dict['0']['recall'],knn_smoteenn_class_rpt_scaled_dict['0']['recall'],
                         logreg_clstrcntrd_class_rpt_scaled_dict['0']['recall'],dectree_clstrcntrd_class_rpt_scaled_dict['0']['recall'],rndfor_clstrcntrd_class_rpt_scaled_dict['0']['recall'],knn_clstrcntrd_class_rpt_scaled_dict['0']['recall']
                           ])
recall_1_scaled_npa = np.array([logreg_class_rpt_scaled_dict['1']['recall'],dectree_class_rpt_scaled_dict['1']['recall'],rndfor_class_rpt_scaled_dict['1']['recall'],knn_class_rpt_scaled_dict['1']['recall'],
                         logreg_rndovr_class_rpt_scaled_dict['1']['recall'],dectree_rndovr_class_rpt_scaled_dict['1']['recall'],rndfor_rndovr_class_rpt_scaled_dict['1']['recall'],knn_rndovr_class_rpt_scaled_dict['1']['recall'],
                         logreg_rndunder_class_rpt_scaled_dict['1']['recall'],dectree_rndunder_class_rpt_scaled_dict['1']['recall'],rndfor_rndunder_class_rpt_scaled_dict['1']['recall'],knn_rndunder_class_rpt_scaled_dict['1']['recall'],
                         logreg_smote_class_rpt_scaled_dict['1']['recall'],dectree_smote_class_rpt_scaled_dict['1']['recall'],rndfor_smote_class_rpt_scaled_dict['1']['recall'],knn_smote_class_rpt_scaled_dict['1']['recall'],
                         logreg_smoteenn_class_rpt_scaled_dict['1']['recall'],dectree_smoteenn_class_rpt_scaled_dict['1']['recall'],rndfor_smoteenn_class_rpt_scaled_dict['1']['recall'],knn_smoteenn_class_rpt_scaled_dict['1']['recall'],
                          logreg_clstrcntrd_class_rpt_scaled_dict['1']['recall'],dectree_clstrcntrd_class_rpt_scaled_dict['1']['recall'],rndfor_clstrcntrd_class_rpt_scaled_dict['1']['recall'],knn_clstrcntrd_class_rpt_scaled_dict['1']['recall']
                           ])

f1_score_0_scaled_npa = np.array([logreg_class_rpt_scaled_dict['0']['f1-score'],dectree_class_rpt_scaled_dict['0']['f1-score'],rndfor_class_rpt_scaled_dict['0']['f1-score'],knn_class_rpt_scaled_dict['0']['f1-score'],
                           logreg_rndovr_class_rpt_scaled_dict['0']['f1-score'],dectree_rndovr_class_rpt_scaled_dict['0']['f1-score'],rndfor_rndovr_class_rpt_scaled_dict['0']['f1-score'],knn_rndovr_class_rpt_scaled_dict['0']['f1-score'],
                           logreg_rndunder_class_rpt_scaled_dict['0']['f1-score'],dectree_rndunder_class_rpt_scaled_dict['0']['f1-score'],rndfor_rndunder_class_rpt_scaled_dict['0']['f1-score'],knn_rndunder_class_rpt_scaled_dict['0']['f1-score'],
                           logreg_smote_class_rpt_scaled_dict['0']['f1-score'],dectree_smote_class_rpt_scaled_dict['0']['f1-score'],rndfor_smote_class_rpt_scaled_dict['0']['f1-score'],knn_smote_class_rpt_scaled_dict['0']['f1-score'],
                           logreg_smoteenn_class_rpt_scaled_dict['0']['f1-score'],dectree_smoteenn_class_rpt_scaled_dict['0']['f1-score'],rndfor_smoteenn_class_rpt_scaled_dict['0']['f1-score'],knn_smoteenn_class_rpt_scaled_dict['0']['f1-score'],
                           logreg_clstrcntrd_class_rpt_scaled_dict['0']['f1-score'],dectree_clstrcntrd_class_rpt_scaled_dict['0']['f1-score'],rndfor_clstrcntrd_class_rpt_scaled_dict['0']['f1-score'],knn_clstrcntrd_class_rpt_scaled_dict['0']['f1-score']
                           ])
f1_score_1_scaled_npa = np.array([logreg_class_rpt_scaled_dict['1']['f1-score'],dectree_class_rpt_scaled_dict['1']['f1-score'],rndfor_class_rpt_scaled_dict['1']['f1-score'],knn_class_rpt_scaled_dict['1']['f1-score'],
                           logreg_rndovr_class_rpt_scaled_dict['1']['f1-score'],dectree_rndovr_class_rpt_scaled_dict['1']['f1-score'],rndfor_rndovr_class_rpt_scaled_dict['1']['f1-score'],knn_rndovr_class_rpt_scaled_dict['1']['f1-score'],
                           logreg_rndunder_class_rpt_scaled_dict['1']['f1-score'],dectree_rndunder_class_rpt_scaled_dict['1']['f1-score'],rndfor_rndunder_class_rpt_scaled_dict['1']['f1-score'],knn_rndunder_class_rpt_scaled_dict['1']['f1-score'],
                           logreg_smote_class_rpt_scaled_dict['1']['f1-score'],dectree_smote_class_rpt_scaled_dict['1']['f1-score'],rndfor_smote_class_rpt_scaled_dict['1']['f1-score'],knn_smote_class_rpt_scaled_dict['1']['f1-score'],
                           logreg_smoteenn_class_rpt_scaled_dict['1']['f1-score'],dectree_smoteenn_class_rpt_scaled_dict['1']['f1-score'],rndfor_smoteenn_class_rpt_scaled_dict['1']['f1-score'],knn_smoteenn_class_rpt_scaled_dict['1']['f1-score'],
                           logreg_clstrcntrd_class_rpt_scaled_dict['1']['f1-score'],dectree_clstrcntrd_class_rpt_scaled_dict['1']['f1-score'],rndfor_clstrcntrd_class_rpt_scaled_dict['1']['f1-score'],knn_clstrcntrd_class_rpt_scaled_dict['1']['f1-score']
                           ])

supervised_learning_summary_df = pd.DataFrame(list(zip(accuracy_npa,accuracy_scaled_npa,precision_0_npa,precision_1_npa,precision_0_scaled_npa,precision_1_scaled_npa,recall_0_npa,recall_1_npa,recall_0_scaled_npa,recall_1_scaled_npa,f1_score_0_npa,f1_score_1_npa,f1_score_0_scaled_npa,f1_score_1_scaled_npa)))

columns = [('Accuracy-unscaled',' '),('Accuracy-scaled',' '),('Precision-unscaled','0'),('Precision-unscaled','1'),('Precision-scaled','0'),('Precision-scaled','1'),('Recall-unscaled','0'),('Recall-unscaled','1'),('Recall-scaled','0'),('Recall-scaled','1'),('F1-Score-unscaled','0'),('F1-Score-unscaled','1'),('F1-Score-scaled','0'),('F1-Score-scaled','1')]
supervised_learning_summary_df.columns = pd.MultiIndex.from_tuples(columns)    

indexes = [('Original','Logistic Regression'),('Original','Decision Tree'),('Original','Random Forest'),('Original','KNN'),
           ('Rand Ovr Sampled','Logistic Regression'),('Rand Ovr Sampled','Decision Tree'),('Rand Ovr Sampled','Random Forest'),('Rand Ovr Sampled','KNN'),
           ('Rand Under Sampled','Logistic Regression'),('Rand Under Sampled','Decision Tree'),('Rand Under Sampled','Random Forest'),('Rand Under Sampled','KNN'),
           ('SMOTE Sampled','Logistic Regression'),('SMOTE Sampled','Decision Tree'),('SMOTE Sampled','Random Forest'),('SMOTE Sampled','KNN'),
           ('SMOTEENN Sampled','Logistic Regression'),('SMOTEENN Sampled','Decision Tree'),('SMOTEENN Sampled','Random Forest'),('SMOTEENN Sampled','KNN'),
           ('Cluster Centroid Sampled','Logistic Regression'),('Cluster Centroid Sampled','Decision Tree'),('Cluster Centroid Sampled','Random Forest'),('Cluster Centroid Sampled','KNN')
          ]
supervised_learning_summary_df.index = pd.MultiIndex.from_tuples(indexes)       

display(supervised_learning_summary_df)


# In[ ]:


# Accuracy Comparison DataFrame

accuracy_df = supervised_learning_summary_df[['Accuracy-unscaled','Accuracy-scaled']]
accuracy_df.columns = [['Accuracy-unscaled','Accuracy-scaled']]
accuracy_unstack_df = accuracy_df.unstack()

print("Accuracy for different ML Models / Scaling / Sampling:")
display(accuracy_unstack_df) 


# In[ ]:


# Accuracy Comparison Barplot

X = ['Original','Random Oversample','Random Undersample','Cluster Centrol', 'SMOTE','SMOTEEN',]

logreg_npa = np.array(accuracy_unstack_df['Accuracy-unscaled']['Logistic Regression'])
dectree_npa = np.array(accuracy_unstack_df['Accuracy-unscaled']['Decision Tree'])
rndfor_npa = np.array(accuracy_unstack_df['Accuracy-unscaled']['Random Forest'])
knn_npa = np.array(accuracy_unstack_df['Accuracy-unscaled']['KNN'])

logreg_scaled_npa = np.array(accuracy_unstack_df['Accuracy-scaled']['Logistic Regression'])
dectree_scaled_npa = np.array(accuracy_unstack_df['Accuracy-scaled']['Decision Tree'])
rndfor_scaled_npa = np.array(accuracy_unstack_df['Accuracy-scaled']['Random Forest'])
knn_scaled_npa = np.array(accuracy_unstack_df['Accuracy-scaled']['KNN'])

plt.figure(figsize=(10, 9))

X_axis = np.arange(len(X))
  
sep = 1/16
plt.bar(X_axis - 5*sep, logreg_npa, sep, label = "Logistic Regression (Unscaled)")
plt.bar(X_axis - 4*sep, logreg_scaled_npa, sep, label = "Logistic Regression (Scaled)")
plt.bar(X_axis - 2*sep, dectree_npa, sep, label = "Decistion Tree  (Unscaled)")
plt.bar(X_axis - 1*sep, dectree_scaled_npa, sep, label = "Decistion Tree (Scaled)")
plt.bar(X_axis + 1*sep, rndfor_npa, sep, label = "Random Forest  (Unscaled)")
plt.bar(X_axis + 2*sep, rndfor_scaled_npa, sep, label = "Random Forest (Scaled)")
plt.bar(X_axis + 4*sep, knn_npa, sep, label = "KNN  (Unscaled)")
plt.bar(X_axis + 5*sep, knn_scaled_npa, sep, label = "KNN (Scaled)")

max_y_lim = max(supervised_learning_summary_df[('Accuracy-unscaled',' ')]) + .005
min_y_lim = min(supervised_learning_summary_df[('Accuracy-unscaled',' ')]) - .01
plt.ylim(min_y_lim, max_y_lim)

plt.xticks(X_axis, X, rotation = 90)
plt.xlabel("Sampling Type")
plt.ylabel("Accuracy")
plt.title("Accuracy for Various ML Models / Scaling / Sampling")
plt.legend()

plt.show()


# In[ ]:


# Precision (Value=0) Comparison DataFrame 

precision_0_df = supervised_learning_summary_df[[('Precision-unscaled','0'),('Precision-scaled','0')]]
precision_0_df.columns = [['Precision-unscaled-0','Precision-scaled-0']]
precision_0_unstack_df = precision_0_df.unstack()

print("Precision/0 for different ML Models / Scaling / Sampling:")
display(precision_0_unstack_df) 


# In[ ]:


# Precision (Value=0) Comparison Barplot

X = ['Original','Random Oversample','Random Undersample','Cluster Centrol', 'SMOTE','SMOTEEN',]

logreg_npa = np.array(precision_0_unstack_df['Precision-unscaled-0']['Logistic Regression'])
dectree_npa = np.array(precision_0_unstack_df['Precision-unscaled-0']['Decision Tree'])
rndfor_npa = np.array(precision_0_unstack_df['Precision-unscaled-0']['Random Forest'])
knn_npa = np.array(precision_0_unstack_df['Precision-unscaled-0']['KNN'])

logreg_scaled_npa = np.array(precision_0_unstack_df['Precision-scaled-0']['Logistic Regression'])
dectree_scaled_npa = np.array(precision_0_unstack_df['Precision-scaled-0']['Decision Tree'])
rndfor_scaled_npa = np.array(precision_0_unstack_df['Precision-scaled-0']['Random Forest'])
knn_scaled_npa = np.array(precision_0_unstack_df['Precision-scaled-0']['KNN'])

plt.figure(figsize=(10, 9))

X_axis = np.arange(len(X))
  
sep = 1/16
plt.bar(X_axis - 5*sep, logreg_npa, sep, label = "Logistic Regression (Unscaled)")
plt.bar(X_axis - 4*sep, logreg_scaled_npa, sep, label = "Logistic Regression (Scaled)")
plt.bar(X_axis - 2*sep, dectree_npa, sep, label = "Decistion Tree  (Unscaled)")
plt.bar(X_axis - 1*sep, dectree_scaled_npa, sep, label = "Decistion Tree (Scaled)")
plt.bar(X_axis + 1*sep, rndfor_npa, sep, label = "Random Forest  (Unscaled)")
plt.bar(X_axis + 2*sep, rndfor_scaled_npa, sep, label = "Random Forest (Scaled)")
plt.bar(X_axis + 4*sep, knn_npa, sep, label = "KNN  (Unscaled)")
plt.bar(X_axis + 5*sep, knn_scaled_npa, sep, label = "KNN (Scaled)")

max_y_lim = max(supervised_learning_summary_df[('Precision-unscaled','0')]) + .006
min_y_lim = min(supervised_learning_summary_df[('Precision-unscaled','0')]) - .01
plt.ylim(min_y_lim, max_y_lim)
plt.xticks(X_axis, X, rotation = 90)
plt.xlabel("Sampling Type")
plt.ylabel("Precision")
plt.title("Precision/0 for Various ML Models / Scaling / Sampling")
plt.legend()

plt.show()


# In[ ]:


# Precision (Value=1) Comparison DataFrame 

precision_1_df = supervised_learning_summary_df[[('Precision-unscaled','1'),('Precision-scaled','1')]]
precision_1_df.columns = [['Precision-unscaled-1','Precision-scaled-1']]
precision_1_unstack_df = precision_1_df.unstack()

print("Precision/1 for different ML Models / Scaling / Sampling:")
display(precision_1_unstack_df) 


# In[ ]:


# Precision (Value=1) Comparison Barplot

X = ['Original','Random Oversample','Random Undersample','Cluster Centrol', 'SMOTE','SMOTEEN',]

logreg_npa = np.array(precision_1_unstack_df['Precision-unscaled-1']['Logistic Regression'])
dectree_npa = np.array(precision_1_unstack_df['Precision-unscaled-1']['Decision Tree'])
rndfor_npa = np.array(precision_1_unstack_df['Precision-unscaled-1']['Random Forest'])
knn_npa = np.array(precision_1_unstack_df['Precision-unscaled-1']['KNN'])

logreg_scaled_npa = np.array(precision_1_unstack_df['Precision-scaled-1']['Logistic Regression'])
dectree_scaled_npa = np.array(precision_1_unstack_df['Precision-scaled-1']['Decision Tree'])
rndfor_scaled_npa = np.array(precision_1_unstack_df['Precision-scaled-1']['Random Forest'])
knn_scaled_npa = np.array(precision_1_unstack_df['Precision-scaled-1']['KNN'])

plt.figure(figsize=(10, 9))

X_axis = np.arange(len(X))
  
sep = 1/16
plt.bar(X_axis - 5*sep, logreg_npa, sep, label = "Logistic Regression (Unscaled)")
plt.bar(X_axis - 4*sep, logreg_scaled_npa, sep, label = "Logistic Regression (Scaled)")
plt.bar(X_axis - 2*sep, dectree_npa, sep, label = "Decistion Tree  (Unscaled)")
plt.bar(X_axis - 1*sep, dectree_scaled_npa, sep, label = "Decistion Tree (Scaled)")
plt.bar(X_axis + 1*sep, rndfor_npa, sep, label = "Random Forest  (Unscaled)")
plt.bar(X_axis + 2*sep, rndfor_scaled_npa, sep, label = "Random Forest (Scaled)")
plt.bar(X_axis + 4*sep, knn_npa, sep, label = "KNN  (Unscaled)")
plt.bar(X_axis + 5*sep, knn_scaled_npa, sep, label = "KNN (Scaled)")

max_y_lim = max(supervised_learning_summary_df[('Precision-unscaled','1')]) + .025
min_y_lim = min(supervised_learning_summary_df[('Precision-unscaled','1')]) - .01
plt.ylim(min_y_lim, max_y_lim)

plt.xticks(X_axis, X, rotation = 90)
plt.xlabel("Sampling Type")
plt.ylabel("Precision")
plt.title("Precision/1 for Various ML Models / Scaling / Sampling")
plt.legend()

plt.show()


# In[ ]:


# Recall (Value=0) Comparison DataFrame 

recall_0_df = supervised_learning_summary_df[[('Recall-unscaled','0'),('Recall-scaled','0')]]
recall_0_df.columns = [['Recall-unscaled-0','Recall-scaled-0']]
recall_0_unstack_df = recall_0_df.unstack()

print("Recall/0 for different ML Models / Scaling / Sampling:")
display(recall_0_unstack_df) 


# In[ ]:


# Recall (Value=0) Comparison Barplot

X = ['Original','Random Oversample','Random Undersample','Cluster Centrol', 'SMOTE','SMOTEEN',]

logreg_npa = np.array(recall_0_unstack_df['Recall-unscaled-0']['Logistic Regression'])
dectree_npa = np.array(recall_0_unstack_df['Recall-unscaled-0']['Decision Tree'])
rndfor_npa = np.array(recall_0_unstack_df['Recall-unscaled-0']['Random Forest'])
knn_npa = np.array(recall_0_unstack_df['Recall-unscaled-0']['KNN'])

logreg_scaled_npa = np.array(recall_0_unstack_df['Recall-scaled-0']['Logistic Regression'])
dectree_scaled_npa = np.array(recall_0_unstack_df['Recall-scaled-0']['Decision Tree'])
rndfor_scaled_npa = np.array(recall_0_unstack_df['Recall-scaled-0']['Random Forest'])
knn_scaled_npa = np.array(recall_0_unstack_df['Recall-scaled-0']['KNN'])

plt.figure(figsize=(10, 9))

X_axis = np.arange(len(X))
  
sep = 1/16
plt.bar(X_axis - 5*sep, logreg_npa, sep, label = "Logistic Regression (Unscaled)")
plt.bar(X_axis - 4*sep, logreg_scaled_npa, sep, label = "Logistic Regression (Scaled)")
plt.bar(X_axis - 2*sep, dectree_npa, sep, label = "Decistion Tree  (Unscaled)")
plt.bar(X_axis - 1*sep, dectree_scaled_npa, sep, label = "Decistion Tree (Scaled)")
plt.bar(X_axis + 1*sep, rndfor_npa, sep, label = "Random Forest  (Unscaled)")
plt.bar(X_axis + 2*sep, rndfor_scaled_npa, sep, label = "Random Forest (Scaled)")
plt.bar(X_axis + 4*sep, knn_npa, sep, label = "KNN  (Unscaled)")
plt.bar(X_axis + 5*sep, knn_scaled_npa, sep, label = "KNN (Scaled)")

max_y_lim = max(supervised_learning_summary_df[('Recall-unscaled','0')]) + .005
min_y_lim = min(supervised_learning_summary_df[('Recall-unscaled','0')]) - .01
plt.ylim(min_y_lim, max_y_lim)

plt.xticks(X_axis, X, rotation = 90)
plt.xlabel("Sampling Type")
plt.ylabel("Recall")
plt.title("Recall/0 for Various ML Models / Scaling / Sampling")
plt.legend()

plt.show()


# In[ ]:


# Recall (Value=1) Comparison DataFrame 

recall_1_df = supervised_learning_summary_df[[('Recall-unscaled','1'),('Recall-scaled','1')]]
recall_1_df.columns = [['Recall-unscaled-1','Recall-scaled-1']]
recall_1_unstack_df = recall_1_df.unstack()

print("Recall/1 for different ML Models / Scaling / Sampling:")
display(recall_1_unstack_df) 


# In[ ]:


# Recall (Value=1) Comparison Barplot

X = ['Original','Random Oversample','Random Undersample','Cluster Centrol', 'SMOTE','SMOTEEN',]

logreg_npa = np.array(recall_1_unstack_df['Recall-unscaled-1']['Logistic Regression'])
dectree_npa = np.array(recall_1_unstack_df['Recall-unscaled-1']['Decision Tree'])
rndfor_npa = np.array(recall_1_unstack_df['Recall-unscaled-1']['Random Forest'])
knn_npa = np.array(recall_1_unstack_df['Recall-unscaled-1']['KNN'])

logreg_scaled_npa = np.array(recall_1_unstack_df['Recall-scaled-1']['Logistic Regression'])
dectree_scaled_npa = np.array(recall_1_unstack_df['Recall-scaled-1']['Decision Tree'])
rndfor_scaled_npa = np.array(recall_1_unstack_df['Recall-scaled-1']['Random Forest'])
knn_scaled_npa = np.array(recall_1_unstack_df['Recall-scaled-1']['KNN'])

plt.figure(figsize=(10, 9))

X_axis = np.arange(len(X))
  
sep = 1/16
plt.bar(X_axis - 5*sep, logreg_npa, sep, label = "Logistic Regression (Unscaled)")
plt.bar(X_axis - 4*sep, logreg_scaled_npa, sep, label = "Logistic Regression (Scaled)")
plt.bar(X_axis - 2*sep, dectree_npa, sep, label = "Decistion Tree  (Unscaled)")
plt.bar(X_axis - 1*sep, dectree_scaled_npa, sep, label = "Decistion Tree (Scaled)")
plt.bar(X_axis + 1*sep, rndfor_npa, sep, label = "Random Forest  (Unscaled)")
plt.bar(X_axis + 2*sep, rndfor_scaled_npa, sep, label = "Random Forest (Scaled)")
plt.bar(X_axis + 4*sep, knn_npa, sep, label = "KNN  (Unscaled)")
plt.bar(X_axis + 5*sep, knn_scaled_npa, sep, label = "KNN (Scaled)")

max_y_lim = max(supervised_learning_summary_df[('Recall-unscaled','1')]) + .055
min_y_lim = min(supervised_learning_summary_df[('Recall-unscaled','1')]) - .01
plt.ylim(min_y_lim, max_y_lim)

plt.xticks(X_axis, X, rotation = 90)
plt.xlabel("Sampling Type")
plt.ylabel("Recall")
plt.title("Recall/1 for Various ML Models / Scaling / Sampling")
plt.legend()

plt.show()


# In[ ]:


# F1-Score (Value=0) Comparison DataFrame 

f1_score_0_df = supervised_learning_summary_df[[('F1-Score-unscaled','0'),('F1-Score-scaled','0')]]
f1_score_0_df.columns = [['F1-Score-unscaled-0','F1-Score-scaled-0']]
f1_score_0_unstack_df = f1_score_0_df.unstack()

print("F1-Score/0 for different ML Models / Scaling / Sampling:")
display(f1_score_0_unstack_df) 


# In[ ]:


# F1-Score (Value=0) Comparison Barplot

X = ['Original','Random Oversample','Random Undersample','Cluster Centrol', 'SMOTE','SMOTEEN',]

logreg_npa = np.array(f1_score_0_unstack_df['F1-Score-unscaled-0']['Logistic Regression'])
dectree_npa = np.array(f1_score_0_unstack_df['F1-Score-unscaled-0']['Decision Tree'])
rndfor_npa = np.array(f1_score_0_unstack_df['F1-Score-unscaled-0']['Random Forest'])
knn_npa = np.array(f1_score_0_unstack_df['F1-Score-unscaled-0']['KNN'])

logreg_scaled_npa = np.array(f1_score_0_unstack_df['F1-Score-scaled-0']['Logistic Regression'])
dectree_scaled_npa = np.array(f1_score_0_unstack_df['F1-Score-scaled-0']['Decision Tree'])
rndfor_scaled_npa = np.array(f1_score_0_unstack_df['F1-Score-scaled-0']['Random Forest'])
knn_scaled_npa = np.array(f1_score_0_unstack_df['F1-Score-scaled-0']['KNN'])

plt.figure(figsize=(10, 9))

X_axis = np.arange(len(X))
  
sep = 1/16
plt.bar(X_axis - 5*sep, logreg_npa, sep, label = "Logistic Regression (Unscaled)")
plt.bar(X_axis - 4*sep, logreg_scaled_npa, sep, label = "Logistic Regression (Scaled)")
plt.bar(X_axis - 2*sep, dectree_npa, sep, label = "Decistion Tree  (Unscaled)")
plt.bar(X_axis - 1*sep, dectree_scaled_npa, sep, label = "Decistion Tree (Scaled)")
plt.bar(X_axis + 1*sep, rndfor_npa, sep, label = "Random Forest  (Unscaled)")
plt.bar(X_axis + 2*sep, rndfor_scaled_npa, sep, label = "Random Forest (Scaled)")
plt.bar(X_axis + 4*sep, knn_npa, sep, label = "KNN  (Unscaled)")
plt.bar(X_axis + 5*sep, knn_scaled_npa, sep, label = "KNN (Scaled)")

max_y_lim = max(supervised_learning_summary_df[('F1-Score-unscaled','0')]) + .005
min_y_lim = min(supervised_learning_summary_df[('F1-Score-unscaled','0')]) - .01
plt.ylim(min_y_lim, max_y_lim)

plt.xticks(X_axis, X, rotation = 90)
plt.xlabel("Sampling Type")
plt.ylabel("F1-Score")
plt.title("F1-Score/0 for Various ML Models / Scaling / Sampling")
plt.legend()

plt.show()


# In[ ]:


# F1-Score (Value=1) Comparison DataFrame 

f1_score_1_df = supervised_learning_summary_df[[('F1-Score-unscaled','1'),('F1-Score-scaled','1')]]
f1_score_1_df.columns = [['F1-Score-unscaled-1','F1-Score-scaled-1']]
f1_score_1_unstack_df = f1_score_1_df.unstack()

print("F1-Score/1 for different ML Models / Scaling / Sampling:")
display(f1_score_1_unstack_df)  


# In[ ]:


# F1-Score (Value=1) Comparison Barplot

X = ['Original','Random Oversample','Random Undersample','Cluster Centrol', 'SMOTE','SMOTEEN',]

logreg_npa = np.array(f1_score_1_unstack_df['F1-Score-unscaled-1']['Logistic Regression'])
dectree_npa = np.array(f1_score_1_unstack_df['F1-Score-unscaled-1']['Decision Tree'])
rndfor_npa = np.array(f1_score_1_unstack_df['F1-Score-unscaled-1']['Random Forest'])
knn_npa = np.array(f1_score_1_unstack_df['F1-Score-unscaled-1']['KNN'])

logreg_scaled_npa = np.array(f1_score_1_unstack_df['F1-Score-scaled-1']['Logistic Regression'])
dectree_scaled_npa = np.array(f1_score_1_unstack_df['F1-Score-scaled-1']['Decision Tree'])
rndfor_scaled_npa = np.array(f1_score_1_unstack_df['F1-Score-scaled-1']['Random Forest'])
knn_scaled_npa = np.array(f1_score_1_unstack_df['F1-Score-scaled-1']['KNN'])

plt.figure(figsize=(10, 9))

X_axis = np.arange(len(X))
  
sep = 1/16
plt.bar(X_axis - 5*sep, logreg_npa, sep, label = "Logistic Regression (Unscaled)")
plt.bar(X_axis - 4*sep, logreg_scaled_npa, sep, label = "Logistic Regression (Scaled)")
plt.bar(X_axis - 2*sep, dectree_npa, sep, label = "Decistion Tree  (Unscaled)")
plt.bar(X_axis - 1*sep, dectree_scaled_npa, sep, label = "Decistion Tree (Scaled)")
plt.bar(X_axis + 1*sep, rndfor_npa, sep, label = "Random Forest  (Unscaled)")
plt.bar(X_axis + 2*sep, rndfor_scaled_npa, sep, label = "Random Forest (Scaled)")
plt.bar(X_axis + 4*sep, knn_npa, sep, label = "KNN  (Unscaled)")
plt.bar(X_axis + 5*sep, knn_scaled_npa, sep, label = "KNN (Scaled)")

max_y_lim = max(supervised_learning_summary_df[('F1-Score-unscaled','1')]) + .03
min_y_lim = min(supervised_learning_summary_df[('F1-Score-unscaled','1')]) - .01
plt.ylim(min_y_lim, max_y_lim)

plt.xticks(X_axis, X, rotation = 90)
plt.xlabel("Sampling Type")
plt.ylabel("F1-Score")
plt.title("F1-Score/1 for Various ML Models / Scaling / Sampling")
plt.legend()

plt.show()


# In[ ]:





# In[ ]:




