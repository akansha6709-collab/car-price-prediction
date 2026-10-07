import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    r"C:\Users\Akansha Singh\Downloads\car data (1).csv"
)

print("Dataset loaded successfully!")


# =========================
# REMOVE DUPLICATES
# =========================

df = df.drop_duplicates()


# =========================
# FEATURES AND TARGET
# =========================

X = df[
    [
        "Year",
        "Present_Price",
        "Fuel_Type",
        "Kms_Driven"
    ]
]

y = df["Selling_Price"]


# =========================
# PREPROCESSING
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "fuel",
            OneHotEncoder(handle_unknown="ignore"),
            ["Fuel_Type"]
        )
    ],
    remainder="passthrough"
)


# =========================
# MODEL
# =========================

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# TRAIN MODEL
# =========================

model.fit(X_train, y_train)


# =========================
# PREDICTION
# =========================

predictions = model.predict(X_test)


# =========================
# EVALUATION
# =========================

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("Model trained successfully!")

print("MAE:", round(mae, 2))

print("R2 Score:", round(r2, 2))


# =========================
# SAVE MODEL
# =========================

Path("model").mkdir(
    exist_ok=True
)

joblib.dump(
    model,
    "model/car_price_model.pkl"
)

print("Model saved successfully!")