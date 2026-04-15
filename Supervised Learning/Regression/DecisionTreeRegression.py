import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('../../Datasets/concrete_data.csv')

x = df.drop(['concrete_compressive_strength'],axis =1)
y = df['concrete_compressive_strength']

categorical_cols = x.select_dtypes(include=['object']).columns
numeric_cols = x.select_dtypes(include = ['number']).columns

categorical_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transfomer = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('Standardscaler', StandardScaler())
])

preprocess = ColumnTransformer(transformers = [
    ('num', numeric_transfomer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

x_processed = preprocess.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_processed, y, test_size = 0.2, random_state=42)

model = DecisionTreeRegressor(max_depth = 5, min_samples_split=10, random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R2 : ",r2)
print("mse : ",mse)