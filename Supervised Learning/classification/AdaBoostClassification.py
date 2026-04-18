import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

df = pd.read_csv('../../Datasets/heart.csv')

x = df.drop(['target'],axis =1)
y = df['target']

categorical_cols = x.select_dtypes(include = ['object']).columns
numeric_cols = x.select_dtypes(include =['number']).columns

categorical_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown = 'ignore'))
])

numeric_tranformer = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('standardscaler', StandardScaler())
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer(transformers= [
    ('num', numeric_tranformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

x_train_processed = preprocess.fit_transform(x_train)
x_test_processed = preprocess.transform(x_test)

base_model = DecisionTreeClassifier(max_depth =1)
model = AdaBoostClassifier(estimator=base_model, n_estimators=100, learning_rate=0.5, random_state=42)

model.fit(x_train_processed, y_train)

y_pred = model.predict(x_test_processed)

accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred, average='weighted')

print("accuracy : ",accuracy)
print("f1-score : ",f1score)