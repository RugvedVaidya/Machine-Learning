# pca - principal component analysis

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.decomposition import PCA

df = pd.read_csv('../../Dataset/bank.csv')

x = df.copy()

categorical_cols = x.select_dtypes(include = ['object']).columns
numeric_cols = x.select_dtypes(include = ['number']).columns

categorical_transformer = Pipeline({
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
})

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocess = ColumnTransformer(transformers = [
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

x_processed = preprocess.fit_transform(x)

model = PCA(n_components=2)

x_pca = model.fit_transform(x_processed)

pca_df = pd.DataFrame(
    x_pca,
    columns =['Principal_component_1', 'Principal_component_2']
)

print("explained vairance ratio : ",model.explained_variance_ratio_)
print("total variance retained : ", np.sum(model.explained_variance_ratio_))