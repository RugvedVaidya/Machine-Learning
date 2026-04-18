import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report 

df = pd.read_csv('../../Datasets/Titanic-Dataset.csv')

x = df.drop(['Survived'],axis=1)
y = df['Survived']

categorical_cols = x.select_dtypes(include =['object']).columns
numeric_cols = x.select_dtypes(include=['number']).columns

categorical_transfer = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('standardscaler', StandardScaler())
])

preprocess = ColumnTransformer(transformers=[
    ('nunm', numeric_transformer, numeric_cols),
    ('cat', categorical_transfer, categorical_cols)
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train_processed = preprocess.fit_transform(x_train)
x_test_processed = preprocess.transform(x_test)

model = LogisticRegression(solver='liblinear')

model.fit(x_train_processed, y_train) 

y_pred = model.predict(x_test_processed)

accuracy = accuracy_score(y_test, y_pred) 
classification_repo = classification_report(y_test, y_pred)

print("accuracy : ",accuracy)
print("classificaion report : ",classification_repo)