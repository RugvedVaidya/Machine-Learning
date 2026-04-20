# lda - linear discriminant analysis

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

df = pd.read_csv('../../Datasets/WineQT.csv')

x = df.drop(['quality'],axis =1)
y = df['quality']

categorical_cols = x.select_dtypes(include = ['object']).columns
numeric_cols = x.select_dtypes(include = ['number']).columns

categorical_transformer = Pipeline([ 
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocess = ColumnTransformer(transformers = [
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

x_processed = preprocess.fit_transform(x)

model = LinearDiscriminantAnalysis(n_components=2)

x_lda = model.fit_transform(x_processed,y)

lda_df = pd.DataFrame(x_lda, columns=['lda_1', 'lda_2'])

lda_df['quality'] = y

print("shape before lda : ",x.shape)
print("shape after lda : ",x_lda.shape)

