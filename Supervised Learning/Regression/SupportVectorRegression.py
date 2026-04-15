import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR 

df = pd.read_csv('../../Datasets/diabetes.csv')

x = df.drop(['Outcome'],axis=1)
y = df['Outcome']

categorical_cols = x.select_dtypes(include = ['object']).columns 
numeric_cols = x.select_dtypes(include = ['number']).columns

categorical_transformer = Pipeline([
    ('impute', SimpleImputer(strategy = 'most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy = 'mean')),
    ('standardscaler', StandardScaler())
])

preprocess = ColumnTransformer(transformers = [
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

x_processed = preprocess.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_processed, y, test_size=0.2, random_state = 42)

model = SVR(kernel = 'rbf')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R2 score : ", r2)
print("MSE : ",mse)
