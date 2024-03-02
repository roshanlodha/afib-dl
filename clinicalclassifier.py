import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("loading and cleaning data...")
vandy = pd.read_csv("/home/lodhar/afib-dl/data/vanderbilt_ct_phenotype_2-14-23.csv")
ccf = pd.read_excel("/home/lodhar/afib-dl/data/CT_demographic_w_ccf_variables 20231102 hsk3.xlsx")
#ccf_temp = pd.read_excel("./data/CCF_CT_demographic.xlsx")
#ccf_temp = ccf_temp[ccf_temp['imageBeforeSurgery'] == "Yes"]


relavent_vandy_columns = ['age_ablation', 'gender', 'race', 'ethnicity', 'pt_height', 'weight', 
                          'htn', 'diabetes', 'cad', 'stroke_tia',
                          'lvef_any_modality', 'paroxsymal', 'long_standing_persistent', 
                          'recurrence']


relavent_ccf_columns = ['Age', 'Gender', 'Race', 'Ethnicity', 'Height', 'Weight',
                        'htn_ablation', 'dm_ablation', 'cad_ablation', 'HxStroke', 'HxTia', 
                        'lvef_ablation', 'af_type_paroxysmal', 'af_type_persistent', 
                        'af_recur']


ccf = ccf[relavent_ccf_columns]
ccf['Hospital'] = "CCF"
vandy = vandy[relavent_vandy_columns]
vandy['Hospital'] = "Vanderbilt"

# make CCF look like Vandy
def combine_HxStroke_HxTia(row):
    if np.isnan(row['HxStroke']) and np.isnan(row['HxTia']):
        return np.nan
    elif row['HxStroke'] == 1 or row['HxTia'] == 1:
        return 1
    else:
        return 0

ccf['HxStrokeTia'] = ccf.apply(combine_HxStroke_HxTia, axis=1)

position = ccf.columns.get_loc("HxStroke") #
ccf.insert(position, 'HxStrokeTia', ccf.pop('HxStrokeTia'))
ccf = ccf.drop(columns = ['HxStroke', 'HxTia'])
ccf.head()


#ccf['image_id'] = ccf['image_id'].astype('object')
#vandy['study_id'] = vandy['study_id'].astype('object')

ccf['Age'] = ccf['Age'].astype('float64')
ccf['Height'] = ccf['Age'].astype('float64')
ccf['Weight'] = ccf['Age'].astype('float64')

def float_to_boolean(value):
    if np.isnan(value):
        return None
    else:
        return bool(value)
    
float_cols = ['htn_ablation', 'dm_ablation', 'cad_ablation', 'HxStrokeTia', 'af_type_paroxysmal', 'af_type_persistent']
#ccf[float_cols] = ccf[float_cols].applymap(float_to_boolean)
#ccf[float_cols] = ccf[float_cols].astype('boolean')

int_to_bool_cols = ['htn', 'diabetes', 'cad', 'stroke_tia', 'paroxsymal', 'long_standing_persistent']
#vandy[int_to_bool_cols] = vandy[int_to_bool_cols].astype('boolean')
vandy[int_to_bool_cols] = vandy[int_to_bool_cols].astype('float64')


gender_map = {0: 'Female', 1: 'Male', 2: 'Unknown'}
vandy['gender'] = vandy['gender'].map(gender_map)

race_map = {
    0: 'American Indian/Alaska Native',
    1: 'Asian',
    2: 'Black or African American',
    3: 'Native Hawaiian or Other Pacific Islander',
    4: 'White',
    5: 'Other',
    6: 'Declined/Prefer not to answer'
}
vandy['race'] = vandy['race'].map(race_map)

ethnicity_map = {
    0: 'Hispanic or Latino',
    1: 'Not Hispanic or Latino',
    2: 'Declined/Prefer not to answer'
}
vandy['ethnicity'] = vandy['ethnicity'].map(ethnicity_map)

vandy.columns = ccf.columns

# combine CCF and Vandy
af_recur_df = pd.concat([ccf, vandy], axis=0)
#af_recur_df = vandy

X = af_recur_df.drop(columns = ['af_recur'])
y = af_recur_df['af_recur']

def one_hot_encode(original_dataframe, feature_to_encode):
    original_dataframe[feature_to_encode] = original_dataframe[feature_to_encode].astype(str)
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

features_to_one_hot_encode = ['Gender', 'Race', 'Ethnicity', 'Hospital']
for feature in features_to_one_hot_encode:
    X = one_hot_encode(X, feature)

print("imputing missing values...")
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# identify variables with missing data
missing_vars = ['htn_ablation', 'dm_ablation', 'cad_ablation', 'HxStrokeTia',
                'lvef_ablation', 'af_type_paroxysmal', 'af_type_persistent']

# identify variables to use as predictors
predictor_vars = ['Age', 'Height', 'Weight', 'Gender_Female', 'Gender_Male', 'Hospital_CCF', 'Hospital_Vanderbilt',
                  'Race_American Indian/Alaska Native', 'Race_Asian', 'Race_Black or African American', 'Race_Declined/Prefer not to answer', 'Race_White',
                 'Ethnicity_Declined/Prefer not to answer', 'Ethnicity_Hispanic or Latino', 'Ethnicity_Not Hispanic or Latino']

# fit regression model using Bayesian Ridge
imp = IterativeImputer(estimator = BayesianRidge())

# impute missing values
imputed_data = imp.fit_transform(X[predictor_vars + missing_vars])

# substitute imputed values for missing values
X[missing_vars] = imputed_data[:, -len(missing_vars):]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # this analysis keeps outliers as they are clinically meaningful
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("modeling...")

from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
param_grid = dict(n_neighbors=list(range(1, X_test.shape[0])))
grid = GridSearchCV(knn, param_grid, cv=10, scoring = 'roc_auc', return_train_score=False)
knn = grid.fit(X_train, y_train)
knn.best_params_

RocCurveDisplay.from_estimator(knn.best_estimator_, X_test, y_test)
plt.savefig("./knnROC.png")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rfc = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
rfc = grid.fit(X_train, y_train)

RocCurveDisplay.from_estimator(rfc.best_estimator_, X_test, y_test)
plt.savefig("./rfcROC.png")

feature_importances = rfc.best_estimator_.feature_importances_

# Create a dictionary mapping feature names to their importances
importance_dict = dict(zip(ccf.columns, feature_importances))

# Print feature importances
print("model importances:")
for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance}")
