# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.feature_selection import SelectKBest, f_regression, chi2, VarianceThreshold, mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from itertools import product
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path
import random
import skfeature
import statsmodels.api as sm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
pd.set_option("display.max_columns", 500)

############################################################################################################

df_test = pd.read_csv('newborn_test.csv')

############################################################################################################
###### TEST DATASET #################
############################################################################################################

df_test['mother_body_mass_index'] = np.where(df_test['mother_body_mass_index'].isnull(), ((df_test['mother_delivery_weight'] - df_test['mother_weight_gain']) / (df_test['mother_height'] ** 2)) * 703, df_test['mother_body_mass_index'])
NA_mother_BMI = df_test['mother_body_mass_index'].isna().sum()
df_test['mother_body_mass_index'] = df_test['mother_body_mass_index'].clip(lower=15, upper=40)
mean_BMI = df_test['mother_body_mass_index'].mean()
df_test['mother_body_mass_index'] = df_test['mother_body_mass_index'].fillna(mean_BMI)
df_test.loc[:, 'mother_marital_status'] = df_test['mother_marital_status'].fillna('unknown')
df_test['mother_delivery_weight'] = np.where(
    df_test['mother_delivery_weight'].isnull() & np.isfinite(df_test['mother_height']) & np.isfinite(df_test['mother_weight_gain']) & np.isfinite(df_test['mother_body_mass_index']),
    (df_test['mother_height'] * (df_test['mother_body_mass_index'] / 703)) + df_test['mother_weight_gain'],
    df_test['mother_delivery_weight']
)
median_mother_delivery_weight = df_test['mother_delivery_weight'].median()
df_test['mother_delivery_weight'] = df_test['mother_delivery_weight'].fillna(median_mother_delivery_weight)
mean_mother_height = df_test['mother_height'].mean()
median_mother_height = df_test['mother_height'].median()
df_test['mother_height'] = np.where(
    df_test['mother_height'].isnull() & np.isfinite(df_test['mother_body_mass_index']) & np.isfinite(df_test['mother_weight_gain']),
    np.sqrt((df_test['mother_delivery_weight'] - df_test['mother_weight_gain']) / (df_test['mother_body_mass_index'] / 703)),
    df_test['mother_height']
)
df_test['mother_height'] = df_test['mother_height'].clip(upper=55)
df_test['mother_height'] = df_test['mother_height'].fillna(mean_mother_height)
mean_mother_weight_gain = df_test['mother_weight_gain'].mean()
median_mother_weight_gain = df_test['mother_weight_gain'].median()
df_test['weight_gain_ratio'] = df_test['mother_weight_gain'] / df_test['mother_delivery_weight']
weight_gain_ratio_mean = df_test['weight_gain_ratio'].mean()
df_test['mother_weight_gain'] = np.where(df_test['mother_weight_gain'].isnull(), df_test['mother_delivery_weight'] * weight_gain_ratio_mean, df_test['mother_weight_gain'])
df_test = df_test.drop('weight_gain_ratio', axis=1)
mean_father_age = df_test['father_age'].mean()
median_father_age = df_test['father_age'].median()
df_test['father_age'] = df_test['father_age'].fillna(mean_father_age)
mean_cigarettes_before_pregnancy = df_test['cigarettes_before_pregnancy'].mean()
median_cigarettes_before_pregnancy = df_test['cigarettes_before_pregnancy'].median()
df_test['cigarettes_before_pregnancy'] = df_test['cigarettes_before_pregnancy'].fillna(median_cigarettes_before_pregnancy)
df_test['prenatal_care_month'] = df_test['prenatal_care_month'].replace(99, 0)
mean_number_prenatal_visits = df_test['number_prenatal_visits'].mean()
df_test['number_prenatal_visits'] = df_test['number_prenatal_visits'].fillna(mean_number_prenatal_visits)
one_hot_encoded_mother_race = pd.get_dummies(df_test['mother_race'], prefix='race')
df_test = pd.concat([df_test, one_hot_encoded_mother_race], axis=1)
df_test = df_test.replace({True: 1, False: 0})
df_test = df_test.drop('mother_race', axis=1)
one_hot_encoded = pd.get_dummies(df_test['mother_marital_status'], prefix='marital_status')
one_hot_encoded = one_hot_encoded.rename(columns={
    'marital_status_1': 'married',
    'marital_status_2.0': 'not_married',
    'marital_status_unknown': 'unknown_married'
})
df_test = pd.concat([df_test, one_hot_encoded], axis=1)
df_test = df_test.replace({True: 1, False: 0})
df_test = df_test.drop('mother_marital_status', axis=1)
one_hot_encoded_newborn_gender = pd.get_dummies(df_test['newborn_gender'])
one_hot_encoded_newborn_gender = one_hot_encoded_newborn_gender.rename(columns={'F': 'baby_gender_female', 'M': 'baby_gender_male'})
df_test = pd.concat([df_test, one_hot_encoded_newborn_gender], axis=1)
df_test = df_test.replace({True: 1, False: 0})
df_test = df_test.drop('newborn_gender', axis=1)
one_hot_encoded_cesarean = pd.get_dummies(df_test['previous_cesarean'])
one_hot_encoded_cesarean = one_hot_encoded_cesarean.rename(columns={'N': 'no_cesarean', 'Y': 'yes_cesarean', 'U':'other_delivery'})
df_test = pd.concat([df_test, one_hot_encoded_cesarean], axis=1)
df_test = df_test.replace({True: 1, False: 0})
df_test = df_test.drop('previous_cesarean', axis=1)
df_test['newborn_weight'] = 0


############################################################################################################
###### TRAIN DATASET #################
############################################################################################################
df_train = pd.read_csv('newborn_train.csv')

# data cleaning

df_train['mother_body_mass_index'] = np.where(df_train['mother_body_mass_index'].isnull(), ((df_train['mother_delivery_weight'] - df_train['mother_weight_gain']) / (df_train['mother_height'] ** 2)) * 703, df_train['mother_body_mass_index'])
mean_mother_body_mass_index = df_train['mother_body_mass_index'].mean()
median_mother_body_mass_index = df_train['mother_body_mass_index'].median()
df_train = df_train[(df_train['mother_body_mass_index'] >= 15) & (df_train['mother_body_mass_index'] <= 40)]

df_train.loc[:, 'mother_marital_status'] = df_train['mother_marital_status'].fillna('unknown')

mean_mother_delivery_weight = df_train['mother_delivery_weight'].mean()
median_mother_delivery_weight = df_train['mother_delivery_weight'].median()

df_train['mother_delivery_weight'] = np.where(
    df_train['mother_delivery_weight'].isnull() & np.isfinite(df_train['mother_height']) & np.isfinite(df_train['mother_weight_gain']) & np.isfinite(df_train['mother_body_mass_index']),
    (df_train['mother_height'] * (df_train['mother_body_mass_index'] / 703)) + df_train['mother_weight_gain'],
    df_train['mother_delivery_weight']
)
df_train['mother_delivery_weight'] = df_train['mother_delivery_weight'].fillna(median_mother_delivery_weight)

df_train = df_train[df_train['mother_height'] >= 55]
mean_mother_height = df_train['mother_height'].mean()
median_mother_height = df_train['mother_height'].median()
df_train['mother_height'] = np.where(
    df_train['mother_height'].isnull() & np.isfinite(df_train['mother_body_mass_index']) & np.isfinite(df_train['mother_weight_gain']),
    np.sqrt((df_train['mother_delivery_weight'] - df_train['mother_weight_gain']) / (df_train['mother_body_mass_index'] / 703)),
    df_train['mother_height']
)
mean_mother_height = df_train['mother_height'].mean()
median_mother_height = df_train['mother_height'].median()
df_train['mother_height'] = df_train['mother_height'].fillna(mean_mother_height)

df_train = df_train[df_train['mother_weight_gain'] != 0]
mean_mother_weight_gain = df_train['mother_weight_gain'].mean()
median_mother_weight_gain = df_train['mother_weight_gain'].median()
df_train['weight_gain_ratio'] = df_train['mother_weight_gain'] / df_train['mother_delivery_weight']
weight_gain_ratio_mean = df_train['weight_gain_ratio'].mean()
df_train['mother_weight_gain'] = np.where(df_train['mother_weight_gain'].isnull(), df_train['mother_delivery_weight'] * weight_gain_ratio_mean, df_train['mother_weight_gain'])
df_train = df_train.drop('weight_gain_ratio', axis=1)

mean_father_age = df_train['father_age'].mean()
median_father_age = df_train['father_age'].median()
df_train['father_age'].describe()
df_train['father_age'] = df_train['father_age'].fillna(mean_father_age)

mean_cigarettes_before_pregnancy = df_train['cigarettes_before_pregnancy'].mean()
median_cigarettes_before_pregnancy = df_train['cigarettes_before_pregnancy'].median()
df_train['cigarettes_before_pregnancy'] = df_train['cigarettes_before_pregnancy'].fillna(median_cigarettes_before_pregnancy)

df_train['prenatal_care_month'] = df_train['prenatal_care_month'].replace(99, 0)
mean_number_prenatal_visits = df_train['number_prenatal_visits'].mean()
median_number_prenatal_visits = df_train['number_prenatal_visits'].median()
df_train['number_prenatal_visits'] = df_train['number_prenatal_visits'].fillna(mean_number_prenatal_visits)

one_hot_encoded_mother_race = pd.get_dummies(df_train['mother_race'], prefix='race')
df_train = pd.concat([df_train, one_hot_encoded_mother_race], axis=1)
df_train = df_train.replace({True: 1, False: 0})
df_train = df_train.drop('mother_race', axis=1)

one_hot_encoded = pd.get_dummies(df_train['mother_marital_status'], prefix='marital_status')
one_hot_encoded = one_hot_encoded.rename(columns={
    'marital_status_1': 'married',
    'marital_status_2.0': 'not_married',
    'marital_status_unknown': 'unknown_married'
})
df_train = pd.concat([df_train, one_hot_encoded], axis=1)
df_train = df_train.replace({True: 1, False: 0})
df_train = df_train.drop('mother_marital_status', axis=1)

one_hot_encoded_newborn_gender = pd.get_dummies(df_train['newborn_gender'])
one_hot_encoded_newborn_gender = one_hot_encoded_newborn_gender.rename(columns={'F': 'baby_gender_female', 'M': 'baby_gender_male'})
df_train = pd.concat([df_train, one_hot_encoded_newborn_gender], axis=1)
df_train = df_train.replace({True: 1, False: 0})
df_train = df_train.drop('newborn_gender', axis=1)

one_hot_encoded_cesarean = pd.get_dummies(df_train['previous_cesarean'])
one_hot_encoded_cesarean = one_hot_encoded_cesarean.rename(columns={'N': 'no_cesarean', 'Y': 'yes_cesarean', 'U':'other_delivery'})
df_train = pd.concat([df_train, one_hot_encoded_cesarean], axis=1)
df_train = df_train.replace({True: 1, False: 0})
df_train = df_train.drop('previous_cesarean', axis=1)



##############################################################################################
##################################            MODELLING            ###########################
##############################################################################################

X_train = df_train.drop(['newborn_weight'], axis=1)
y_train = df_train['newborn_weight']

X_test = df_test.drop("newborn_weight", axis=1)

best_params = {'max_depth': 6, 'learning_rate': 0.009, 'n_estimators': 1000, 'booster': 'gbtree',
               'gamma': 7, 'subsample': 0.6, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7,
               'reg_alpha': 33, 'reg_lambda': 3}

xgb_model = XGBRegressor(random_state=3, **best_params)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

df_test_pred = df_test.copy()
df_test_pred["newborn_weight_pred"] = y_pred
df_test_pred.to_csv('newborn_test_pred_full.csv', index=False)
df_test_pred[["newborn_weight_pred"]].to_csv('newborn_test_pred_values.csv', index=False)





