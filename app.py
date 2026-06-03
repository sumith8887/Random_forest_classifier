import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ---------------------------
# App config
# ---------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)

st.title("Heart Disease Prediction using Random Forest Classifier")
st.caption("Streamlit app built for the heart disease dataset.")

DATA_FILE = "data/heart.csv"
TARGET_COL = "HeartDisease"

NUMERIC_FEATURES = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------
# Load data
# ---------------------------
@st.cache_data
def load_data():
    if not Path(DATA_FILE).exists():
        return None
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------
# Train model
# ---------------------------
@st.cache_resource
def train_model(df):
    missing = [c for c in ALL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError("Missing expected columns: " + ", ".join(missing))

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].astype(int)

    # Make sure numeric columns are numeric
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Make sure categorical columns are strings
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=None
            ))
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, digits=3),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape
    }

    return model, results, df


# ---------------------------
# Main flow
# ---------------------------
df = load_data()

if df is None:
    st.error("Dataset file not found. Please place it at: data/heart.csv")
    st.stop()

try:
    model, results, df = train_model(df)
except Exception as exc:
    st.error(str(exc))
    st.stop()


# ---------------------------
# Layout
# ---------------------------
left, right = st.columns([1.05, 0.95])

with left:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    with st.expander("Show dataset summary"):
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        st.write(df.describe(include="all"))

with right:
    st.subheader("Model Performance")
    st.metric("Accuracy", f"{results['accuracy']:.2%}")
    st.write("Train shape:", results["train_shape"])
    st.write("Test shape:", results["test_shape"])
    st.text(results["report"])
    st.write("Confusion Matrix")
    st.write(
        pd.DataFrame(
            results["confusion_matrix"],
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"]
        )
    )

st.divider()
st.subheader("Predict Heart Disease")

c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
    sex = st.selectbox("Sex", sorted(df["Sex"].dropna().astype(str).unique().tolist()))
    chest_pain = st.selectbox(
        "ChestPainType",
        sorted(df["ChestPainType"].dropna().astype(str).unique().tolist())
    )
    resting_bp = st.number_input("RestingBP", min_value=0, max_value=300, value=120, step=1)
    cholesterol = st.number_input("Cholesterol", min_value=0, max_value=700, value=200, step=1)

with c2:
    fasting_bs = st.selectbox("FastingBS", sorted(df["FastingBS"].dropna().unique().tolist()))
    resting_ecg = st.selectbox("RestingECG", sorted(df["RestingECG"].dropna().astype(str).unique().tolist()))
    max_hr = st.number_input("MaxHR", min_value=0, max_value=250, value=150, step=1)
    exercise_angina = st.selectbox(
        "ExerciseAngina",
        sorted(df["ExerciseAngina"].dropna().astype(str).unique().tolist())
    )
    oldpeak = st.number_input("Oldpeak", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    st_slope = st.selectbox("ST_Slope", sorted(df["ST_Slope"].dropna().astype(str).unique().tolist()))

input_df = pd.DataFrame(
    [[
        age,
        resting_bp,
        cholesterol,
        fasting_bs,
        max_hr,
        oldpeak,
        sex,
        chest_pain,
        resting_ecg,
        exercise_angina,
        st_slope
    ]],
    columns=ALL_FEATURES
)

# Keep the same dtypes as training
for col in NUMERIC_FEATURES:
    input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

for col in CATEGORICAL_FEATURES:
    input_df[col] = input_df[col].astype(str)

if st.button("Predict"):
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    if prediction == 1:
        st.error(f"Prediction: Heart disease likely. Risk score: {probability:.2%}")
    else:
        st.success(f"Prediction: Heart disease unlikely. Risk score: {probability:.2%}")

st.caption("Educational demo only. Not medical advice.")