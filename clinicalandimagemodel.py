#!/usr/bin/env python
# coding: utf-8
from util import predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

training_dir_file = open("/home/lodhar/afib-dl/data/vandy_trainingdirs.txt", "r") 
training_dirs = training_dir_file.read() 
training_dir_file.close()
training_dirs = training_dirs.split("\n") 

vandy = pd.read_csv("/home/lodhar/afib-dl/data/vanderbilt_ct_phenotype_2-14-23.csv")
vandy['nifti_dir'] = '/home/lodhar/afib-dl/nifti/vandy/' + vandy['study_id'].astype(str) + '.nii.gz'
vandy['is_training'] = vandy['nifti_dir'].isin(training_dirs)
vandy

from sklearn.preprocessing import OneHotEncoder

outcomes = [col for col in vandy.columns if 'recur' in col.lower()]
valve_dx_cols = [col for col in vandy.columns if 'type_valve_dx' in col.lower()]
ablation_cols = [col for col in vandy.columns if 'ablation' in col.lower()]
    
vandy = vandy.drop(valve_dx_cols + ablation_cols, axis = 1).drop(['study_id', 'mri_ct', 'nifti_dir', 'date_of_recur', 'time_to_recur', 'la_any_modality'], axis = 1)

def one_hot_encode(original_dataframe, feature_to_encode):
    original_dataframe[feature_to_encode] = original_dataframe[feature_to_encode].astype(str)
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

features_to_one_hot_encode = ['race', 'ethnicity']
for feature in features_to_one_hot_encode:
    vandy = one_hot_encode(vandy, feature)

features_to_binarize = vandy.select_dtypes('int64').columns
vandy

train = vandy[vandy['is_training']].drop(['is_training'], axis = 1).reset_index(drop=True)
test = vandy[~vandy['is_training']].drop(['is_training'], axis = 1).reset_index(drop=True)
X_train = train.drop(['recurrence'], axis = 1)
y_train = train['recurrence']
X_test = test.drop(['recurrence'], axis = 1)
y_test = test['recurrence']

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay

for column in X_test.columns:
    for idx, value in X_test[column].items():
        try:
            float(value)
        except ValueError:
            print(f"Row index: {idx}, Column: {column}, Value: {value}")

clinical_cols = [col for col in X_train.columns if 'activation' not in col]
scaler = StandardScaler()
scaler.fit(X_train[clinical_cols])
X_train[clinical_cols] = scaler.fit_transform(X_train[clinical_cols])
X_test[clinical_cols] = scaler.transform(X_test[clinical_cols])

knn = KNeighborsClassifier()
param_grid = dict(n_neighbors=list(range(1, min(X_test.shape[0], X_train.shape[0])//2)))
grid = GridSearchCV(knn, param_grid, cv=10, scoring = 'roc_auc', return_train_score=False)
knn = grid.fit(X_train, y_train)

RocCurveDisplay.from_estimator(knn.best_estimator_, X_test, y_test)
plt.savefig("./tROC.png")