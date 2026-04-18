import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

df_train = pd.read_csv('../../Datasets/employee_attrition_train.csv')
df_test = pd.read_csv('../../Datasets/employee_attrition_test.csv')

x_train = df_train.drop(['Attrition'],axis=1)
y_train = df_train['Attrition']

x_test = df_test.drop(['Attrition'],axis=1)
y_test = df_test['Attrition']

categorical_cols = x_train.select_dtypes(include=['object']).columns
numeric_cols = x_train.select_dtypes(include=['number']).columns

categorical_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('standardscaler', StandardScaler())
])

preprocess = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_cols),
    ('num', numeric_transformer, numeric_cols)
])

x_train_processed = preprocess.fit_transform(x_train)
x_test_processed = preprocess.transform(x_test)

model = CatBoostClassifier(verbose=0)
model.fit(x_train_processed, y_train)

y_pred = model.predict(x_test_processed)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_mat)
