import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

df = pd.read_csv('../../Datasets/Mall_Customers.csv')

x = df.copy()

categorical_cols = x.select_dtypes(include = ['object']).columns 
numeric_cols = x.select_dtypes(include=['number']).columns

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown = 'ignore'))
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

model = DBSCAN(eps = 0.8, min_samples =5)

clusters = model.fit_predict(x_processed)

df['cluster'] = clusters

#DBSCAN may create noise points labelled as -1
valid_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
if(valid_clusters > 1):
    score = silhouette_score(x_processed, clusters)
    print("score : ",score)
else :
    print("silhouette score cant be determined as only one cluster in present")
    
print("cluster counts : ", df['cluster'].value_counts())

plt.figure(figsize=(8, 6))

plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['cluster']
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('DBSCAN Clustering')

plt.show()