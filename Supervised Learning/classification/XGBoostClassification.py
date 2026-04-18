import numpy as np 
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_csv('../../Datasets/mushrooms.csv')

x = df.drop(['class'],axis=1)
y = df['class']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

categorical_cols = x.select_dtypes(include=['object']).columns
numeric_cols = x.select_dtypes(include=['number']).columns

categorical_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown = 'ignore'))
])

numeric_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('standardScaler', StandardScaler())
])

preprocess = ColumnTransformer(transformers =[
    ('cat', categorical_transformer, categorical_cols),
    ('num', numeric_transformer, numeric_cols)
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state =42)

x_train_processed = preprocess.fit_transform(x_train)
x_test_processed = preprocess.transform(x_test)

model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(x_train_processed, y_train)

y_pred = model.predict(x_test_processed)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("accuracy : ",accuracy)
print("confusion_mat : ",confusion_mat)
