## I. IMPORT PACKAGES, LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt



# PATH TO FILE
storedata_filepath = 'D:/02_AI/storedata.csv'
data = pd.read_csv(storedata_filepath)
# print(data.describe())
training_data = data.copy()
print(training_data.columns)

## II. PREPROCESS DATA
# Why is staff number of store 2039 equal to -2???


# print(data.columns)
"""['Town', 'Country', 'Store ID', 'Manager name', 'Staff', 'Floor Space',
             'Window', 'Car park', 'Demographic score', 'Location',
             '40min population', '30 min population', '20 min population',
             '10 min population', 'Store age', 'Clearance space',
             'Competition number', 'Competition score', 'Performance']"""
# Intuitively, Store ID is not relevant to the performance of a store
# Drop Store ID
training_data.drop('Store ID', axis=1, inplace=True)
print(type(training_data))
# print(training_data.columns)

# ENCODE CATEGORICAL DATA
s = (training_data.dtypes == 'object')
object_cols = list(s[s].index)
training_data['Car park'] = training_data['Car park'].replace(['Y','N','Yes', 'No'],[1, 0, 1, 0])
training_data['Country'] = training_data['Country'].replace(['UK','France'],[0, 1])
training_data['Performance'] = training_data['Performance'].replace(['Good', 'Bad'], [1, 0])
y = training_data['Performance']
X = training_data.drop('Performance', axis=1)
y_test = y
X_test = X
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numerical_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

print(numerical_features)

categorical_transformer = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='NA'),
    OneHotEncoder(handle_unknown = 'ignore', sparse=False)
)
numerical_transformer = make_pipeline(
    SimpleImputer(strategy='constant'),
    StandardScaler()
)


preprocessor = make_column_transformer(
    (numerical_transformer, numerical_features),
    (categorical_transformer, categorical_features),
)
# print(X.columns)
def split_data(features, target, train_data_size):

    X_train, X_valid, y_train, y_valid = train_test_split(features, target, stratify=y, train_size=train_data_size, random_state= 0)

    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    return X_train, y_train, X_valid, y_valid


def evaluate_model(X_train, y_train, X_valid, y_valid):
    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X_train, y_train)
    preds =model.predict(X_valid)
    MAE = mean_absolute_error(y_valid, preds)
    print(MAE)
    preds = np.round(model.predict(X_valid))
    print(confusion_matrix(y_valid,preds))
train_sizes = [0.5, 0.67, 0.8]
for train_size in train_sizes:
    X_train, y_train, X_valid, y_valid = split_data(X, y, train_size)
    evaluate_model(X_train, y_train, X_valid, y_valid)
