# Business Problem/Goal of the ML project:

# To Determine whether patient has malignant tumor or benign tumor was the semi-automated and lengthy process in practice. The older process leads to manual intervention in result of threats and challenges like bad accuracy, exra time taken, overwait, overprocessing, duplicasy of works, waiting time in the cycle. To overcome from these challenges and threats using machine learning technique to create Cancer Detection ML model to classify malignant and benign tumor from the given features.

# Tools Setup for the Business Problem:

# Import All required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Collection by Load breast cancer dataset

from sklearn.datasets import load_breast_cancer
df_cancer = load_breast_cancer()

type(df_cancer)

df_cancer.keys()

df_cancer["data"]

df_cancer["target"]

df_cancer["target_names"]

print(df_cancer["DESCR"])

print(df_cancer["feature_names"])

# Data Pre-processing

# Creating DataFrame with the help of Pandas and Numpy

np.append(df_cancer['feature_names'], ['target'])

df_main_cancer = pd.DataFrame(np.c_[df_cancer['data'],df_cancer['target']],
             columns = np.append(df_cancer['feature_names'], ['target']))

df_main_cancer

# Exploratory Data Analysis With the help of Numpy, Pandas, Matplotlib, Seaborn

df_main_cancer.shape

df_main_cancer.head()

df_main_cancer.tail()

df_main_cancer.info()

df_main_cancer.describe()

# Data Visualization with the help of the Matplotlib/ Seaborn

sns.countplot(df_main_cancer['target'])

sns.pairplot(df_main_cancer, hue = 'target', 
             vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'] )

plt.figure(figsize = (20,8))
sns.countplot(df_main_cancer['mean radius'])

plt.figure(figsize=(12,9))
sns.heatmap(df_main_cancer)

# Finding the co-relation between the Features and Target

plt.figure(figsize=(20,20))
sns.heatmap(df_main_cancer.corr(), annot = True, cmap ='coolwarm', linewidths=2)

df_main_cancer.corr()

corr_matrix = df_main_cancer.corr()

corr_matrix["target"].sort_values(ascending=True)

# Process to make data ready for Training and Testing

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

X = df_main_cancer.drop(['target'], axis = 1)
X.head()

y = df_main_cancer['target']
y.head()

y.tail()

# Perform the step of splitting dataset into train data and test data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( X,y, test_size = 0.2, random_state = 100)

print("shape of x_train = ", x_train.shape)
print("shape of y_train = ", y_train.shape)
print("shape of x_test = ", x_test.shape)
print("shape of y_train = ", y_test.shape)

# Step to transform Data with Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(x_train)

x_train_sc = sc.transform(x_train)

x_train_sc

x_test_sc = sc.transform(x_test)

x_test_sc

# Step to Select possible Model's Training and Testing

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# Support Vector Classifier

from sklearn.svm import SVC


svc_classifier = SVC()
svc_classifier.fit(x_train, y_train)
y_pred_SVC = svc_classifier.predict(x_test)
accuracy_score(y_test, y_pred_SVC)

# Support Vector Classifier With StandardScaler

svc_classifier2 = SVC()
svc_classifier2.fit(x_train_sc, y_train)
y_pred_SVC_sc = svc_classifier2.predict(x_test_sc)
accuracy_score(y_test, y_pred_SVC_sc)

# Logistic Regression

from sklearn.linear_model import LogisticRegression


lr_classifier = LogisticRegression(random_state = 100, penalty = 'l2')
lr_classifier.fit(x_train, y_train)
y_pred_lr = lr_classifier.predict(x_test)
accuracy_score(y_test, y_pred_lr)

# LogisticRegression With StandardScaler

lr_classifier2 = LogisticRegression(random_state = 100, penalty = 'l2')
lr_classifier2.fit(x_train_sc, y_train)
y_pred_lr_sc = lr_classifier.predict(x_test_sc)
accuracy_score(y_test, y_pred_lr_sc)

# K – Nearest Neighbor Classifier

from sklearn.neighbors import KNeighborsClassifier



knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(x_train, y_train)
y_pred_knn = knn_classifier.predict(x_test)
accuracy_score(y_test, y_pred_knn)

# K – Nearest Neighbor Classifier With StandardScaler

knn_classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier2.fit(x_train_sc, y_train)
y_pred_knn_sc = knn_classifier2.predict(x_test_sc)
accuracy_score(y_test, y_pred_knn_sc)

# Naive Bayes Classifier

from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)
y_pred_nb = nb_classifier.predict(x_test)
accuracy_score(y_test, y_pred_nb)

# Naive Bayes Classifier With StandardScaler

nb_classifier2 = GaussianNB()
nb_classifier2.fit(x_train_sc, y_train)
y_pred_nb_sc = nb_classifier2.predict(x_test_sc)
accuracy_score(y_test, y_pred_nb_sc)

# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier


dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier.fit(x_train, y_train)
y_pred_dt = dt_classifier.predict(x_test)
accuracy_score(y_test, y_pred_dt)

# Decision Tree Classifier With StandardScaler

dt_classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier2.fit(x_train_sc, y_train)
y_pred_dt_sc = dt_classifier2.predict(x_test_sc)
accuracy_score(y_test, y_pred_dt_sc)

# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier


rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(x_train, y_train)
y_pred_rf = rf_classifier.predict(x_test)
accuracy_score(y_test, y_pred_rf)

# Random Forest Classifier With StandardScaler

rf_classifier2 = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier2.fit(x_train_sc, y_train)
y_pred_rf_sc = rf_classifier2.predict(x_test_sc)
accuracy_score(y_test, y_pred_rf_sc)

# Adaboost Classifier

from sklearn.ensemble import AdaBoostClassifier


adb_classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier.fit(x_train, y_train)
y_pred_adb = adb_classifier.predict(x_test)
accuracy_score(y_test, y_pred_adb)

# Adaboost Classifier With StandardScaler

adb_classifier2 = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier2.fit(x_train_sc, y_train)
y_pred_adb_sc = adb_classifier2.predict(x_test_sc)
accuracy_score(y_test, y_pred_adb_sc)

# XGBoost Classifier

from xgboost import XGBClassifier

xgb_classifier = XGBClassifier()
xgb_classifier.fit(x_train, y_train)
y_pred_xgb = xgb_classifier.predict(x_test)
accuracy_score(y_test, y_pred_xgb)

# XGBoost Classifier with StandardScaler

xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(x_train_sc, y_train)
y_pred_xgb_sc = xgb_classifier2.predict(x_test_sc)
accuracy_score(y_test, y_pred_xgb_sc)

# Use of GridSearch to find the Best Model with best parameters

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

model_params = {
    'SVC': {
        'model': SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    
    'LogisticRegression': {
        'model': LogisticRegression(random_state = 100, penalty = 'l2'),
        'params' : {
        }
    },
    
     'KNeighborsClassifier': {
        'model': KNeighborsClassifier(p=2),
        'params' : {
            'n_neighbors' : [5,7],
        }
    },
    
    'GaussianNB': {
        'model': GaussianNB(priors=None, var_smoothing=1e-09),
        'params' : {
        }
    },
    
    'DecisionTreeClassifier': {
        'model': DecisionTreeClassifier(random_state=51),
        'params' : {
            'criterion' : ['gini', 'entropy'],
        }
    },
        
    'RandomForestClassifier': {
        'model': RandomForestClassifier(criterion='gini', random_state=51),
        'params' : {
            'n_estimators': [10,20],
                      
        }
    },
    
     'AdaBoostClassifier': {
        'model': AdaBoostClassifier(base_estimator=None, algorithm='SAMME.R', random_state=1,),
        'params' : {
            'n_estimators': [1,5,10],
            'learning_rate': [0.1,1.0],
        }
    },
    
    'XGBClassifier': {
        'model': XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1),
        'params' : {
            
        }
        
    },
    
    
}

from sklearn.model_selection import GridSearchCV


import pandas as pd
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df

# Hyperparameter tunning with the help of RandomizedSearchCV

# XGBoost Parameter Tuning Randomized Search

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 
}

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(xgb_classifier, param_distributions=params, scoring= 'roc_auc', n_jobs= -1, verbose= 3)
random_search.fit(x_train, y_train)

random_search.best_params_


random_search.best_estimator_

# Training XGBoost classifier with best parameters

xgb_classifier_pt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7,
              enable_categorical=False, gamma=0.0, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=6,
              min_child_weight=5, missing=np.nan, monotone_constraints='()',
              n_estimators=100, n_jobs=4, num_parallel_tree=1, predictor='auto',
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
              subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None)
 
xgb_classifier_pt.fit(x_train, y_train)
y_pred_xgb_pt = xgb_classifier_pt.predict(x_test)

accuracy_score(y_test, y_pred_xgb_pt)

cm = confusion_matrix(y_test, y_pred_xgb_pt)
plt.title('Heatmap of Confusion Matrix', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()

print(classification_report(y_test, y_pred_xgb_pt))

# Cross Validation for selected Model

from sklearn.model_selection import cross_val_score

cross_validation = cross_val_score(estimator = xgb_classifier_pt, X = x_train_sc, y = y_train, cv = 10)
print("Cross validation of XGBoost model = ",cross_validation)
print("Cross validation of XGBoost model (in mean) = ",cross_validation.mean())

from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_classifier_pt, X = x_train_sc,y = y_train, cv = 10)
print("Cross validation accuracy of XGBoost model = ", cross_validation)
print("\nCross validation mean accuracy of XGBoost model = ", cross_validation.mean())

# Best Model Deployment with the help of Pickle

import pickle

# Dump the best Model with all waights and values in binary

pickle.dump(xgb_classifier_pt, open('breast_cancer_detector.pickle', 'wb'))

# Load the Model with dump file

breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))

# Prediction with the help of Model

y_pred_model = breast_cancer_detector_model.predict(x_test)
 


# Confusion matrix results from the Model

print('Confusion matrix of XGBoost model: \n',confusion_matrix(y_test, y_pred_model),'\n')
 

# Final Accuracy for the Model


print('Accuracy of XGBoost model = ',accuracy_score(y_test, y_pred_model))

# After Model Deployment need to Monitor the Model and maintain necessary retrainings to achive maximum accuracy.

# Conclusion: With the help of ML Model we can determine whether patient has malignant tumor or benign tumor and eliminate the older lengthy processes which was in practice. With the help of ML Model we can reduce the manual intervention and eliminate the threats and challenges like error in prediction, exra time taken, overwait, overprocessing, duplicasy of works, waiting time in the cycle.


