import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

af = pd.read_csv("./data/vanderbilt_ct_phenotype_2-14-23.csv")
outcomes = [col for col in af.columns if 'recur' in col.lower()]
valve_dx_cols = [col for col in af.columns if 'type_valve_dx' in col.lower()]
ablation_cols = [col for col in af.columns if 'ablation' in col.lower()]
af['la_any_modality'] = af['la_any_modality'].replace(".", None).astype('float')
af = af[af['la_any_modality'].notna()]

gaf = af.groupby('recurrence')
af = gaf.apply(lambda x: x.sample(gaf.size().min()).reset_index(drop=True))

X = af.drop(outcomes + valve_dx_cols + ablation_cols, axis = 1).drop(['study_id', 'mri_ct'], axis = 1)
y = af['recurrence'].reset_index(drop=True)

def one_hot_encode(original_dataframe, feature_to_encode):
    original_dataframe[feature_to_encode] = original_dataframe[feature_to_encode].astype(str)
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

features_to_one_hot_encode = ['race', 'ethnicity']
for feature in features_to_one_hot_encode:
    X = one_hot_encode(X, feature)

features_to_binarize = X.select_dtypes('int64').columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # this analysis keeps outliers as they are clinically meaningful
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay

knn = KNeighborsClassifier()
param_grid = dict(n_neighbors=list(range(1, X_test.shape[0])))
grid = GridSearchCV(knn, param_grid, cv=10, scoring = 'roc_auc', return_train_score=False)
knn = grid.fit(X_train, y_train)
knn.best_params_

RocCurveDisplay.from_estimator(knn.best_estimator_, X_test, y_test)
plt.savefig("./figs/clinicalROC.png")

import pickle as pkl
# save the model to disk
filename = 'knn_clinical.sav'
pkl.dump(knn.best_estimator_, open(filename, 'wb'))

#loaded_model = pickle.load(open(filename, 'rb'))