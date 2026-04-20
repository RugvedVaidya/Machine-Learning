# umap = unifrom manifold approximation and projection

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import umap.umap_ as umap

from sklearn.datasets import load_digits
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 

data = load_digits()

df = pd.DataFrame(data.data)

df['target'] = data.target

x = df.drop(['target'],axis =1)
y = df['target']

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

model = umap.UMAP(
    n_components=2,
    n_neighbors = 15,
    min_dist =0.1,
    random_state=42
)

x_umap = model.fit_transform(x_processed)

umap_df = pd.DataFrame(x_umap, columns=['umap_1', 'umap_2'])

umap_df['quality'] = y

print("shape before umap : ",x.shape)
print("shape after umap : ",x_umap.shape)