import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score

# Step 1: Load dataset
df = pd.read_csv('../../Datasets/Mall_Customers.csv')

# Step 2: Select features
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

# Step 7: Estimate bandwidth
bandwidth = estimate_bandwidth(
    x_processed,
    quantile=0.2,
    n_samples=500
)

# Step 8: Train Mean Shift model
model = MeanShift(bandwidth=bandwidth)

clusters = model.fit_predict(x_processed)

# Step 9: Add cluster labels
df['Cluster'] = clusters

# Step 10: Evaluate model
valid_clusters = len(set(clusters))

if valid_clusters > 1:
    score = silhouette_score(x_processed, clusters)
    print("Silhouette Score:", score)
else:
    print("Silhouette Score cannot be calculated because only one cluster was found.")

print("\nNumber of Clusters Found:", valid_clusters)

print("\nCluster Counts:")
print(df['Cluster'].value_counts())

print("\nFirst Few Rows:")
print(df.head())

# Step 11: Visualize clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster']
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Mean Shift Clustering')

plt.show()