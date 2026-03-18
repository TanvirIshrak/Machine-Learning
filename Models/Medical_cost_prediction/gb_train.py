import pandas as pd
import pickle
import numpy as np

# preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# regression
from sklearn.ensemble import GradientBoostingRegressor
# matrics
from sklearn.metrics import r2_score,mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# Load dataset
df = pd.read_csv('insurance.csv')
df.head(10)


# target and feature
x = df.drop('charges', axis=1)
y=df['charges']

# column split
numeric_feature = x.select_dtypes(include=['int64','float64']).columns
categircal_feature = x.select_dtypes(include=['object']).columns


# Preprocessing
#pipeline for numerical
num_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

#pipeline for categorical
cat_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# combining
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer , numeric_feature),
        ('cat', cat_transformer, categircal_feature)
    ]
)

# GradiantBoost Model
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
)

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', gb_model)
])

x_train , x_test, y_train , y_test = train_test_split(
    x,y, test_size=0.2, random_state=42
)
gb_pipeline.fit(x_train, y_train)


# Evaluation
y_pred = gb_pipeline.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")


# Save model (IMPORTANT)
with open("medical_cost_gb_pipeline.pkl", "wb") as f:
    pickle.dump(gb_pipeline, f)

print("✅ Random Forest pipeline saved as medical_cost_gb_pipeline.pkl")
