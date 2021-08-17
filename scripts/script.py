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
data["Feature4"] = data["NumberofOfficers"]**2
data["Feature5"] = data["GrossLoanPortfolio"]**2
data["Feature6"] = data["NumberofActiveBorrowers"]**2
data["Feature7"] = data["NumberofLoansOutstanding"]**2
data["NumberofOfficers"] = data["NumberofOfficers"]/data["NumberofOfficers"].max()
data["GrossLoanPortfolio"] = data["GrossLoanPortfolio"]/data["GrossLoanPortfolio"].max()
data["NumberofActiveBorrowers"] = data["NumberofActiveBorrowers"]/data["NumberofActiveBorrowers"].max()
data["NumberofLoansOutstanding"] = data["NumberofLoansOutstanding"]/data["NumberofLoansOutstanding"].max()
data["ShortTermDelinquency"] = data["ShortTermDelinquency"]/data["ShortTermDelinquency"].max()
data["LongTermDelinquency"] = data["LongTermDelinquency"]/data["LongTermDelinquency"].max()
data["AvgLoanPerBorrower"] = data["AvgLoanPerBorrower"]/data["AvgLoanPerBorrower"].max()
data["CostperBorrower"] = data["CostperBorrower"]/data["CostperBorrower"].max()
data["BorrowersPerLoanOfficer"] = data["BorrowersPerLoanOfficer"]/data["BorrowersPerLoanOfficer"].max()

data["Feature1"] = data["GrossLoanPortfolio"]/data["NumberofOfficers"]
data["Feature2"] = data["NumberofLoansOutstanding"]/data["NumberofOfficers"]
data["Feature3"] = data["GrossLoanPortfolio"]/data["NumberofOfficers"]



data["Feature4"] = data["Feature4"]/data["Feature4"].max()
data["Feature5"] = data["Feature5"]/data["Feature5"].max()
data["Feature6"] = data["Feature6"]/data["Feature6"].max()
data["Feature7"] = data["Feature7"]/data["Feature7"].max()

#Applying Binary Classification
def is_profitable(profit):
    if (profit>0):
        return 1
    else:
        return 0

data["Profit"] = data["Profit"].apply(is_profitable)

count = 0
countp = 0
for x in data["Profit"]:
  if (x==0):
    count+=1
  else:
    countp+=1
print(count)
print(countp)

#Train-test-split & Upsampling
df_train, df_test, xx, xxx = train_test_split(data, yvals, test_size=0.25, random_state =13)

df_0 = df_train[df_train.Profit==0]
df_1 = df_train[df_train.Profit==1]
df_upsampled = resample(df_0,replace=True,n_samples=len(df_1))
df_train = pd.concat([df_1, df_upsampled])
print(df_train)

#Separating data from prediction values
y_train = df_train["Profit"]
df_train = df_train.drop(['Profit'],axis = 1)
y_test = df_test["Profit"]
df_test = df_test.drop(["Profit"],axis = 1)

#Training Logistic Classifier

logistic_model = LogisticRegression()
logistic_model.fit(df_train,y_train)
y_pred = logistic_model.predict(df_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
coefs = logistic_model.coef_
print(coefs)

#Training Neural Network Classifier - for the neural network runs, a validation set was also included
nnet = MLPClassifier(hidden_layer_sizes=(16,16,16), max_iter= 10000000) 
nnet.fit(df_train, y_train)

# Predict using the validation data and optimize neural network dimensions
predictions = nnet.predict(df_val)
predictionset  = []
for x in predictions:
  if(x>=0.5):
    predictionset.append(1)
  else:
    predictionset.append(0)
    

print(accuracy_score(y_val,predictionset))

predictions = nnet.predict(df_test)
predictionset  = []
for x in predictions:
  if(x>=0.5):
    predictionset.append(1)
  else:
    predictionset.append(0)
    

print(accuracy_score(y_test,predictionset))

#Training Decision Tree Classifier
from sklearn import tree

# We'll first specify what model we want, in this case a decision tree
class_dt = tree.DecisionTreeClassifier(max_depth=12)

# We use our previous `X_train` and `y_train` sets to build the model
class_dt.fit(df_train,y_train)
y_pred = class_dt.predict(df_test)
print(accuracy_score(y_test,y_pred))

#Feature Importance
importances = class_dt.feature_importances_
print(importances)
indices = np.argsort(importances)
features = ["NumberofOfficers","GrossLoanPortfolio","NumberofActiveBorrowers","NumberofLoansOutstanding","ShortTermDelinquency","LongTermDelinquency", "AvgLoanPerBorrower","CostperBorrower","BorrowersPerLoanOfficer","Feature1","Feature2","Feature3","Feature4","Feature5","Feature6","Feature7"]
#features = ["NumberofOfficers","GrossLoanPortfolio","NumberofActiveBorrowers","NumberofLoansOutstanding","ShortTermDelinquency","LongTermDelinquency", "AvgLoanPerBorrower","CostperBorrower","BorrowersPerLoanOfficer"]
plt.title('Feature Importances')
j = 9# top j importance
plt.barh(range(j), importances[indices][len(indices)-j:], color='g', align='center')
plt.yticks(range(j), [features[i] for i in indices[len(indices)-j:]])
plt.xlabel('Relative Importance')
plt.show()

#Training Gradient Boosted Tree
import shap
import numpy as np
import tensorflow.keras.backend 


model = xgboost.XGBClassifier().fit(df_train, y_train)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
predictions = model.predict(df_test)
print(predictions[1:10])
print(accuracy_score(y_test,predictions))

#SHAP Importance
explainer = shap.Explainer(model)
shap_values = explainer(df_train)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])
