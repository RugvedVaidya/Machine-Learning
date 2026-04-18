import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv('../../Datasets/Mall_Customers.csv')

x = df.copy()

categorical_cols = x.select_dtypes(include=['object']).columns
numeric_cols = x.select_dtypes(include=['number']).columns

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocess = ColumnTransformer(transformers = [
    ('cat', categorical_transformer, categorical_cols),
    ('num', numeric_transformer, numeric_cols)
])

x_processed = preprocess.fit_transform(x)

plt.figure(figsize=(12, 6))

linked = linkage(x_processed, method = 'ward')

dendrogram(linked)

plt.title('Dendogram')
plt.xlabel('data points')
plt.ylabel('distance')
plt.show()

model = AgglomerativeClustering(
    n_clusters=3,
    linkage = 'ward'
)

clusters = model.fit_predict(x_processed)

df['Cluster'] = clusters

score = silhouette_score(x_processed, clusters)

print("silhouette score : ",score)
print("no of clusters : ",df['Cluster'].value_counts())

plt.figure(figsize=(8, 6))
plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Hierarchical Clustering')

plt.show()