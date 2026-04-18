import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv('../../Datasets/UCI_Credit_Card.csv')

# Separate features and target
x = df.drop(['default.payment.next.month'], axis=1)
y = df['default.payment.next.month']

# Identify categorical and numeric columns
categorical_cols = x.select_dtypes(include=['object']).columns
numeric_cols = x.select_dtypes(include=['number']).columns

# Define transformers
categorical_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('standardscaler', StandardScaler())
])

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Preprocess data
preprocess = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_cols),
    ('num', numeric_transformer, numeric_cols)
])

x_train_processed = preprocess.fit_transform(x_train)
x_test_processed = preprocess.transform(x_test)

# Train Bagging Classifier
base_classifier = DecisionTreeClassifier(random_state=42)
model = BaggingClassifier(estimator=base_classifier, n_estimators=10, random_state=42)
model.fit(x_train_processed, y_train)

# Make predictions
y_pred = model.predict(x_test_processed)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_mat)
print("\nClassification Report:\n", classification_report(y_test, y_pred))