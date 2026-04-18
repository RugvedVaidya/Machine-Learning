import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from lightgbm import LGBMClassifier

df = pd.read_csv('../../Datasets/UCI_Credit_Card.csv') 

x = df.drop(['default.payment.next.month'],axis=1)
y = df['default.payment.next.month']

categorical_cols = x.select_dtypes(include = ['object']).columns
numeric_cols = x.select_dtypes(include =['number']).columns

categorical_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown = 'ignore'))
])

numeric_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('standardscaler', StandardScaler())
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =42)

preprocess = ColumnTransformer(transformers = [
    ('cat', categorical_transformer, categorical_cols),
    ('num', numeric_transformer, numeric_cols)
])

x_train_processed = preprocess.fit_transform(x_train) 
x_test_processed = preprocess.transform(x_test) 

model = LGBMClassifier(force_col_wise = True)
model.fit(x_train_processed, y_train)

y_pred = model.predict(x_test_processed) 

accuracy = accuracy_score(y_test, y_pred) 
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy : ",accuracy)
print("confusion matrix : ",confusion_mat)