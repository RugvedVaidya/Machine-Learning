import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Step 1: Load dataset
data = load_digits()

# Step 2: Create DataFrame
df = pd.DataFrame(data.data)

# Step 3: Add target column
df['target'] = data.target

# Step 4: Separate features and target
x = df.drop('target', axis=1)
y = df['target']

# Step 5: Separate categorical and numerical columns
categorical_cols = x.select_dtypes(include=['object', 'category']).columns
numeric_cols = x.select_dtypes(include=['number']).columns

# Step 6: Create preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

# Step 7: Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Step 8: Preprocess data
x_processed = preprocessor.fit_transform(x)

# Step 9: Define Autoencoder Architecture
input_dim = x_processed.shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))

# Encoder
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
bottleneck = Dense(encoding_dim, activation='linear')(encoded)

# Decoder
decoded = Dense(16, activation='relu')(bottleneck)
decoded = Dense(32, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

# Step 10: Create models
autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder = Model(inputs=input_layer, outputs=bottleneck)

# Step 11: Compile model
autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

# Step 12: Train model
autoencoder.fit(
    x_processed,
    x_processed,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Step 13: Get compressed representation
x_encoded = encoder.predict(x_processed)

# Step 14: Create DataFrame
encoded_df = pd.DataFrame(
    x_encoded,
    columns=['Encoded_Feature_1', 'Encoded_Feature_2']
)

# Step 15: Add target column
encoded_df['target'] = y

print("Shape Before Autoencoder:", x.shape)
print("Shape After Autoencoder:", x_encoded.shape)

print("\nFirst Few Rows:")
print(encoded_df.head())

# Step 16: Visualize compressed representation
plt.figure(figsize=(8, 6))

for target_class in encoded_df['target'].unique():
    temp_df = encoded_df[encoded_df['target'] == target_class]

    plt.scatter(
        temp_df['Encoded_Feature_1'],
        temp_df['Encoded_Feature_2'],
        label=f'Class {target_class}'
    )

plt.xlabel('Encoded Feature 1')
plt.ylabel('Encoded Feature 2')
plt.title('Autoencoder Visualization')
plt.legend()

plt.show()