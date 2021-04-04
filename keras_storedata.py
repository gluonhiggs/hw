## I. IMPORT PACKAGES, LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, confusion_matrix

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
print(X.columns)
def assess_model(X,y, batch_size, epochs_num):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify= y, train_size=0.8)

    X_train = preprocessor.fit_transform(X_train)
    X_valid = preprocessor.transform(X_valid)

    input_shape = [X_train.shape[1]]

    model = keras.Sequential([
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', input_shape = input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])


    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['binary_accuracy'],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        patience=50,
        min_delta=0.0001,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=batch_size,
        epochs=epochs_num,
        callbacks=[early_stopping],
    )

    preds = model.predict(X_valid)
    # print(preds)
    MAE = mean_absolute_error(y_valid, preds)

    y_preds = np.round(model.predict(X_valid))
    conf_m = confusion_matrix(y_valid, y_preds)
    return MAE,  conf_m


batch_sizes = [50, 60, 70, 80]
epochs_nums = [50, 100, 150]
assess_array = []
for batch_size in batch_sizes:
    for epochs_num in epochs_nums:
        MAE, conf_m = assess_model(X,y, batch_size, epochs_num)
        assess_array.append([MAE, conf_m])
print(assess_array)
