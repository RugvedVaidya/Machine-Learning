import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# load dataset
df = pd.read_csv('../../Datasets/Titanic-Dataset.csv')

# feature and target column selection
x = df.drop(['Survived'], axis=1)
y = df['Survived']

# identify column types
categorical_cols = x.select_dtypes(include=['object']).columns
numeric_cols = x.select_dtypes(include=['number']).columns

# preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# apply preprocessing
x_processed = preprocessor.fit_transform(x)

# split dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x_processed, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(x_train, y_train)

# predict
y_pred = model.predict(x_test)
y_pred_rounded = np.round(y_pred).astype(int)  # Round predictions to nearest integer for classification

# evaluation
r2 = r2_score(y_test, y_pred_rounded)
mse = mean_squared_error(y_test, y_pred_rounded)
accuracy = accuracy_score(y_test, y_pred_rounded)
conf_matrix = confusion_matrix(y_test, y_pred_rounded)
report = classification_report(y_test, y_pred_rounded)

print("Accuracy: ", accuracy)
print("Confusion Matrix: \n", conf_matrix)
print("Classification Report: \n", report)
print("R2 Score: ", r2)
print("Mean Squared Error: ", mse)