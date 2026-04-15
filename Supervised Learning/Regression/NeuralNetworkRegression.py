import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('../../Datasets/Housing.csv')

x = df.drop(['price'],axis=1)
y = df['price']

categorical_cols = x.select_dtypes(include=['object']).columns
numeric_cols = x.select_dtypes(include = ['number']).columns

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('standardscaler', StandardScaler())
])

preprocessing = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

x_processed = preprocessing.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_processed, y, test_size=0.2, random_state=42)

model = MLPRegressor(activation='relu', max_iter=1000, random_state=42, solver='adam')

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("r2 : ",r2)
print("mse : ",mse)
