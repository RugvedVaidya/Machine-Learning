import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv('../../Datasets/Fishers maket.csv')

x = df.drop(['Weight'],axis =1)
y = df['Weight']

categorical_cols = x.select_dtypes(include = ['object']).columns
numeric_cols = x.select_dtypes(include=['number']).columns

categorical_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('standardscaler', StandardScaler())
])

preprocess = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_cols),
    ('num', numeric_transformer, numeric_cols)
])

x_processed = preprocess.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_processed, y, test_size=0.2, random_state=42)

model = KNeighborsRegressor(n_neighbors= 5, weights='distance')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("r2 : ",r2)
print("mse : ",mse)