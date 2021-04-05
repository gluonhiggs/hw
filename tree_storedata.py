
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, confusion_matrix, plot_roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


storedata_filepath = 'D:/02_AI/storedata.csv' # path to file
data = pd.read_csv(storedata_filepath)
# print(data.describe())
training_data = data.copy()

## PREPROCESS DATA
# Why is staff number of store 2039 equal to -2???
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
# print("Categorical variables:")
# print(object_cols)
"""['Town', 'Country', 'Manager name', 'Car park', 'Location', 'Performance']"""

training_data['Car park'] = training_data['Car park'].replace(['Y','N','Yes', 'No'],[1, 0, 1, 0])
training_data['Performance'] = training_data['Performance'].replace(['Good', 'Bad'], [1, 0])
training_data['Country'] = training_data['Country'].replace(['UK','France'],[0, 1])
y = training_data['Performance']
X = training_data.drop('Performance', axis=1)
y_test = y
X_test = X

def split_data(X,y,train_size): # def a split_data function so that we can use it multiple times with different values of train_size
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=train_size, stratify= y, random_state=2)
    return X_train, y_train, X_valid, y_valid


def assess_model(X_train_input, X_valid_input, y_train_input, y_valid_input, num_estimators): # use this multiple times with different values of num_estimators (trees)
    model = RandomForestClassifier(n_estimators = num_estimators, random_state=1)
    model.fit(X_train_input, y_train_input)
    preds = model.predict(X_valid_input)
    MAE = mean_absolute_error(y_valid_input, preds)
    y_preds = np.round(preds)
    conf_m = confusion_matrix(y_valid_input, y_preds)
    return MAE, conf_m

MAE_array = [] # list of different MAEs
train_sizes = [0.5, 0.67, 0.8]
for train_size in train_sizes:
    X_train, y_train, X_valid, y_valid = split_data(X,y,train_size)

    object_cols = ['Town', 'Manager name',  'Location']
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
    OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index
    OH_cols_test.index = X_test.index
    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)
    num_X_test = X_test.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
    OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
    # print(type(OH_X_train))
    print('Model with train-set-split size = {}\n'.format(train_size))
    for num_estimators in range(50, 500, 50):
        print('- When number of trees is {}\n'.format(num_estimators))
        MAE, conf_mat = assess_model(OH_X_train, OH_X_valid, y_train, y_valid, num_estimators)
        print('Mean absolute error:\n')
        print(MAE)
        print('Confusion Matrix:\n')
        print(conf_mat)
        MAE_array.append(MAE)
        
# find hyperparameters corresponding to the minimum absolute error
train_size_index = MAE_array.index(min(MAE_array))//len(range(50,500,50))
tree_num_index = MAE_array.index(min(MAE_array))%len(range(50,500,50))
print(MAE_array.index(min(MAE_array)))
print('Min error: {} with train size {} and {} trees.'.format(min(MAE_array), train_sizes[train_size_index], range(50,500,50)[tree_num_index]))






#ROC

X_train, y_train, X_valid, y_valid = split_data(X,y,train_sizes[train_size_index])

object_cols = ['Town', 'Country', 'Manager name',  'Location']
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index
OH_cols_test.index = X_test.index
# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)
num_X_test = X_test.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
# print(type(OH_X_train))
model = RandomForestClassifier(n_estimators = range(50,500,50)[tree_num_index], random_state=1)
model.fit(OH_X_train, y_train)
preds = model.predict(OH_X_valid)
y_probas = model.predict_proba(OH_X_valid)
plot_roc_curve(model, OH_X_valid, y_valid, name='ROC for valid data')
plot_roc_curve(model, OH_X_test, y_test, name='ROC for test data')
plt.show()


