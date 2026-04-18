import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Step 1: Load dataset
df = pd.read_csv('../../Datasets/Mall_Customers.csv')

# Step 2: Select features for clustering
x = df.copy()

# Step 3: Separate categorical and numerical columns
categorical_cols = x.select_dtypes(include=['object']).columns
numeric_cols = x.select_dtypes(include=['number']).columns

# Step 4: Create preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

# Step 5: Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Step 6: Preprocess data
x_processed = preprocessor.fit_transform(x)

# Step 7: Use Elbow Method to find optimal number of clusters
wcss = [] #within cluster sum of squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(x_processed)
    wcss.append(kmeans.inertia_)

# Step 8: Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Step 9: Train K-Means model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = model.fit_predict(x_processed)

# Step 10: Add cluster labels to original dataframe
df['Cluster'] = clusters

# Step 11: Evaluate clustering performance
score = silhouette_score(x_processed, clusters)

print("Silhouette Score:", score)

print("\nCluster Counts:")
print(df['Cluster'].value_counts())

print("\nFirst Few Rows:")
print(df.head())

# Step 12: Visualize clusters using original columns
plt.figure(figsize=(8, 6))

plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Clusters')

plt.show()