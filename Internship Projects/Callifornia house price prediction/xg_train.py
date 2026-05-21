import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_squared_error
from xgboost import XGBRegressor


data = pd.read_csv('housing.csv')

x = data.drop(['median_house_value'],axis=1)
y = data['median_house_value']


# column split
numeric_feature = x.select_dtypes(include=['int64','float64']).columns
categircal_feature = x.select_dtypes(include=['object']).columns

num_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_feature),
        ('cat', cat_transformer, categircal_feature)
    ]
)


xg_model = XGBRegressor(
    n_estimators=100,
    max_depth=10,
    objective='reg:squarederror',
    random_state=42,
    device='cuda',
    tree_method='hist'
)

xg_pipe = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', xg_model)
        ]
    )


x_train , x_test, y_train , y_test = train_test_split(
    x,y, test_size=0.2, random_state=42
)

xg_pipe.fit(x_train,y_train)
y_pred = xg_pipe.predict(x_test)

r2=r2_score(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))

print("R2 Score:", r2)
print("RMSE:", rmse)

with open("house_cost_xg_pipeline.pkl", "wb") as f:
    pickle.dump(xg_pipe, f)

print("✅ XGB pipeline saved as house_cost_xg_pipeline.pkl")