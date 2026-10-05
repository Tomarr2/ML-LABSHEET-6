import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. CREATE DATASET
# --------------------------------------------------

data = {
    "area": [
        1000, 1200, 1500, 1800, 2000,
        900, 1100, 1600, 2200, 2500,
        1300, 1400, 1700, 1900, 2100,
        800, 1250, 1550, 1850, 2300
    ],

    "bedrooms": [
        2, 2, 3, 3, 4,
        2, 2, 3, 4, 4,
        3, 3, 3, 4, 4,
        1, 2, 3, 3, 4
    ],

    "location": [
        "Urban", "Suburban", "Urban", "Urban", "Urban",
        "Rural", "Suburban", "Urban", "Urban", "Urban",
        "Suburban", "Urban", "Suburban", "Urban", "Urban",
        "Rural", "Suburban", "Urban", "Suburban", "Urban"
    ],

    "bathrooms": [
        1, 2, 2, 3, 3,
        1, 2, 2, 3, 4,
        2, 2, 3, 3, 3,
        1, 2, 2, 3, 4
    ],

    "parking": [
        0, 1, 1, 2, 2,
        0, 1, 1, 2, 2,
        1, 1, 2, 2, 2,
        0, 1, 1, 2, 2
    ],

    "age": [
        10, 8, 5, 3, 2,
        15, 10, 6, 4, 2,
        9, 7, 5, 4, 3,
        18, 12, 8, 5, 2
    ],

    "price": [
        4500000, 5500000, 7500000, 9500000, 11500000,
        3000000, 5000000, 8000000, 12500000, 14500000,
        6500000, 7200000, 9000000, 10800000, 13000000,
        2700000, 5200000, 7800000, 9800000, 13500000
    ]
}

df = pd.DataFrame(data)

print("HOUSE PRICE DATASET")
print(df.head())

# --------------------------------------------------
# 2. DATA PREPROCESSING
# --------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

df = df.drop_duplicates()

# --------------------------------------------------
# 3. FEATURES AND TARGET
# --------------------------------------------------

X = df[
    [
        "area",
        "bedrooms",
        "location",
        "bathrooms",
        "parking",
        "age"
    ]
]

y = df["price"]

# --------------------------------------------------
# 4. TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# --------------------------------------------------
# 5. PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_features = [
    "area",
    "bedrooms",
    "bathrooms",
    "parking",
    "age"
]

categorical_features = [
    "location"
]

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)

# --------------------------------------------------
# 6. MODEL 1 - LINEAR REGRESSION
# --------------------------------------------------

linear_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_pipeline.fit(
    X_train,
    y_train
)

linear_prediction = linear_pipeline.predict(
    X_test
)

linear_mae = mean_absolute_error(
    y_test,
    linear_prediction
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_prediction
    )
)

linear_r2 = r2_score(
    y_test,
    linear_prediction
)

print("\n----- LINEAR REGRESSION -----")
print("MAE :", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))
print("R2 Score:", round(linear_r2, 2))

# --------------------------------------------------
# 7. MODEL 2 - RANDOM FOREST
# --------------------------------------------------

random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

random_forest_pipeline.fit(
    X_train,
    y_train
)

rf_prediction = random_forest_pipeline.predict(
    X_test
)

rf_mae = mean_absolute_error(
    y_test,
    rf_prediction
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_prediction
    )
)

rf_r2 = r2_score(
    y_test,
    rf_prediction
)

print("\n----- RANDOM FOREST REGRESSOR -----")
print("MAE :", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R2 Score:", round(rf_r2, 2))

# --------------------------------------------------
# 8. VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    rf_prediction
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 9. SAVE TRAINED MODEL
# --------------------------------------------------

joblib.dump(
    random_forest_pipeline,
    "house_price_model.pkl"
)

print("\nModel saved as:")
print("house_price_model.pkl")

# --------------------------------------------------
# 10. PREDICT NEW HOUSE PRICE
# --------------------------------------------------

new_house = pd.DataFrame({
    "area": [1800],
    "bedrooms": [3],
    "location": ["Urban"],
    "bathrooms": [2],
    "parking": [1],
    "age": [5]
})

predicted_price = random_forest_pipeline.predict(
    new_house
)

print("\nPredicted House Price:")
print(round(predicted_price[0], 2))