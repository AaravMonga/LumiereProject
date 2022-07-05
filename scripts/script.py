#Loading packages
import os              
import numpy as np    
import pandas as pd   
from sklearn.metrics import r2_score   
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import random
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pylab as plt
from sklearn import datasets, ensemble, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import xgboost
import shap

#Reading files
data_path = "Inddata.csv"
y_vals = "indprofit.csv"

data = pd.read_csv(data_path, encoding='latin1')
yvals = pd.read_csv(y_vals)
data.head()

#Processing data

#Handcoding engineered features - arbitrary based on experimentation
data["Feature4"] = data["NumberofOfficers"]**2
data["Feature5"] = data["GrossLoanPortfolio"]**2
data["Feature6"] = data["NumberofActiveBorrowers"]**2
data["Feature7"] = data["NumberofLoansOutstanding"]**2

#Implementing simple 0-1 scale data normalization
data["NumberofOfficers"] = data["NumberofOfficers"]/data["NumberofOfficers"].max()
data["GrossLoanPortfolio"] = data["GrossLoanPortfolio"]/data["GrossLoanPortfolio"].max()
data["NumberofActiveBorrowers"] = data["NumberofActiveBorrowers"]/data["NumberofActiveBorrowers"].max()
data["NumberofLoansOutstanding"] = data["NumberofLoansOutstanding"]/data["NumberofLoansOutstanding"].max()
data["ShortTermDelinquency"] = data["ShortTermDelinquency"]/data["ShortTermDelinquency"].max()
data["LongTermDelinquency"] = data["LongTermDelinquency"]/data["LongTermDelinquency"].max()
data["AvgLoanPerBorrower"] = data["AvgLoanPerBorrower"]/data["AvgLoanPerBorrower"].max()
data["CostperBorrower"] = data["CostperBorrower"]/data["CostperBorrower"].max()
data["BorrowersPerLoanOfficer"] = data["BorrowersPerLoanOfficer"]/data["BorrowersPerLoanOfficer"].max()

#Handcoding meaningful relationships between features that a model would struggle to learn by itself
data["Feature1"] = data["GrossLoanPortfolio"]/data["NumberofOfficers"]
data["Feature2"] = data["NumberofLoansOutstanding"]/data["NumberofOfficers"]
data["Feature3"] = data["GrossLoanPortfolio"]/data["NumberofOfficers"]


#Normalizing engineered features
data["Feature4"] = data["Feature4"]/data["Feature4"].max()
data["Feature5"] = data["Feature5"]/data["Feature5"].max()
data["Feature6"] = data["Feature6"]/data["Feature6"].max()
data["Feature7"] = data["Feature7"]/data["Feature7"].max()

#Applying Binary Classification Mask - simplifying the output to make models more interpretable. The goal here is to interpret, not use, the model in order to obtain insights. 
#While applying a binary mask reduces the complexity of the prediction, it makes the model far more interpretable.
def is_profitable(profit):
    if (profit>0):
        return 1
    else:
        return 0

data["Profit"] = data["Profit"].apply(is_profitable)


#Train-test-split & Upsampling - correcting imbalanced classes which is skewed towards profit
df_train, df_test, xx, xxx = train_test_split(data, yvals, test_size=0.25, random_state =13)

df_0 = df_train[df_train.Profit==0]
df_1 = df_train[df_train.Profit==1]

#upsampling so that number of positive cases = number of negative cases; upsampling comes at the cost of model extrapolability, but the data available is limited in this manner.
df_upsampled = resample(df_0,replace=True,n_samples=len(df_1))
df_train = pd.concat([df_1, df_upsampled])

#Separating data from prediction values
y_train = df_train["Profit"]
df_train = df_train.drop(['Profit'],axis = 1)
y_test = df_test["Profit"]
df_test = df_test.drop(["Profit"],axis = 1)

#Training Baseline Logistic Classifier. We evaluate model performance and make interpretations at several levels of model complexity. Note that for the logistic classifier, only the signs of the coefficients were at all considered in interpretation.
#Numerous papers have discussed that directly using weights of models to argue statistical relationships is flawed.

logistic_model = LogisticRegression()
logistic_model.fit(df_train,y_train)
y_pred = logistic_model.predict(df_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
coefs = logistic_model.coef_



#For more complex models, Decision Tree classifier variations were considered. This was to maximize model interpretability; decision trees learn a hierarchal decision-making process, which mimics the logical reasoning of a human being.
#This makes its decision-making process less arbitrary than other classifiers, such as deep neural networks.
#Training Decision Tree Classifier
from sklearn import tree

# The max-depth was chosen based on experimentation. There was not much tuning of hyperparameters, because the goal of this paper was not to obtain the most accurate predictor, but rather an explainable predictor with a reasonable degree of accuracy.
# This would allow a model simple enough to be interpreted, yet accurate enough for these interpretations to reflect real relationships.
class_dt = tree.DecisionTreeClassifier(max_depth=12)


class_dt.fit(df_train,y_train)
y_pred = class_dt.predict(df_test)

#Feature Importance. The features that contribute to profit were analyzed using feature importances. This is a technique unique to hierarchal decision making processes which measures the entropy change when a node splits; i.e., how well a feature can be used to separate positive and negative examples.
importances = class_dt.feature_importances_
print(importances)
indices = np.argsort(importances)
features = ["NumberofOfficers","GrossLoanPortfolio","NumberofActiveBorrowers","NumberofLoansOutstanding","ShortTermDelinquency","LongTermDelinquency", "AvgLoanPerBorrower","CostperBorrower","BorrowersPerLoanOfficer","Feature1","Feature2","Feature3","Feature4","Feature5","Feature6","Feature7"]
#features = ["NumberofOfficers","GrossLoanPortfolio","NumberofActiveBorrowers","NumberofLoansOutstanding","ShortTermDelinquency","LongTermDelinquency", "AvgLoanPerBorrower","CostperBorrower","BorrowersPerLoanOfficer"]
plt.title('Feature Importances')
j = 9 # top j important features
plt.barh(range(j), importances[indices][len(indices)-j:], color='g', align='center')
plt.yticks(range(j), [features[i] for i in indices[len(indices)-j:]])
plt.xlabel('Relative Importance')
plt.show()

#Training Gradient Boosted Tree. We now move to a more complex decision structure, which uses an ensemble of weak predictors to output a binary prediction.
#Gradient boosting Decision Trees increases structural complexity, which means we must use a more complex metric of feature importance - Shapley values.
#Shapley values arise from a concept in game theory, measuring the marginal contribution of a feature to the average prediction.

import shap
import numpy as np
import tensorflow.keras.backend 

#fitting XGB classifier
model = xgboost.XGBClassifier().fit(df_train, y_train)

# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
predictions = model.predict(df_test)

#print(accuracy_score(y_test,predictions))

#SHAP Importance
explainer = shap.Explainer(model)
shap_values = explainer(df_train)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])
