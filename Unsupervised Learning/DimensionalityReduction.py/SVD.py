#svd - single value decomposition

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD

data = fetch_openml(name='mnist_784', version=1, as_frame=True)

df = data.frame

df['target'] = data.target

x = df.drop('target', axis=1)
y = df['target']

categorical_cols = x.select_dtypes(include=['object', 'category']).columns
numeric_cols = x.select_dtypes(include=['number']).columns

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

x_processed = preprocessor.fit_transform(x)

model = TruncatedSVD(n_components=2, random_state=42)

x_svd = model.fit_transform(x_processed)

svd_df = pd.DataFrame(
    x_svd,
    columns=['SVD_Component_1', 'SVD_Component_2']
)

svd_df['target'] = y

print("Explained Variance Ratio:")
print(model.explained_variance_ratio_)

print("\nTotal Variance Retained:")
print(np.sum(model.explained_variance_ratio_))

print("\nFirst Few Rows:")
print(svd_df.head())

plt.figure(figsize=(8, 6))

for target_class in svd_df['target'].unique():
    temp_df = svd_df[svd_df['target'] == target_class]

    plt.scatter(
        temp_df['SVD_Component_1'],
        temp_df['SVD_Component_2'],
        label=f'Class {target_class}'
    )

plt.xlabel('SVD Component 1')
plt.ylabel('SVD Component 2')
plt.title('SVD Visualization')
plt.legend()

plt.show()