# t-distributed stochastic neighbor embedding

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.manifold import TSNE

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

model = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

x_tsne = model.fit_transform(x_processed)

tsne_df = pd.DataFrame(x_tsne, columns=['tsne_1', 'tsne_2'])

tsne_df['quality'] = y

print("shape before tsne : ",x.shape)
print("shape after tsne : ",x_tsne.shape)