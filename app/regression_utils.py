"""Cleaning, feature engineering and preprocessing for the insurance-claim regression model.

Mirrors REVIEW1_REGRESSION.ipynb (sections A1, B1-B3) so training and the Streamlit app use
exactly the same logic. The sklearn preprocessor is a normal Pipeline/ColumnTransformer, so it
is fitted on the TRAIN split only (no leakage).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "total_claim_amount"

NUMERIC_COLS = ["age", "annual_income_inr", "bmi", "alcohol_units_per_week",
                "has_diabetes", "has_hypertension", "family_history_cardiac",
                "stress_level_score", "sum_insured", "icu_days", "length_of_stay",
                "days_since_policy_start", "total_hospital_days", "icu_ratio",
                "comorbidity_count", "bmi_risk", "claim_month"]
CATEGORICAL_COLS = ["gender", "marital_status", "state", "city", "occupation_type",
                    "tobacco_usage", "physical_activity_level", "diet_type",
                    "policy_type", "hospital_tier", "room_category",
                    "cashless_claim", "agent_channel"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
FLAG_COLS = ["has_diabetes", "has_hypertension"]
PLAIN_NUM = [c for c in NUMERIC_COLS if c not in FLAG_COLS]


def clean_currency(s):
    """'₹151,678' -> 151678.0 (also handles plain numbers)."""
    if pd.isna(s):
        return np.nan
    s = str(s).replace("\u20b9", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_income(s):
    """Handles the Indian 'LPA' notation: 1 lakh = 100,000."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if "LPA" in s.upper():
        return float(s.upper().replace("LPA", "").strip()) * 100_000
    return clean_currency(s)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["total_claim_amount", "hospital_bill_amount"]:
        if c in df:
            df[c] = df[c].apply(clean_currency)
    df["annual_income_inr"] = df["annual_income_inr"].apply(clean_income)
    df["gender"] = (df["gender"].str.strip().str.lower()
                    .map({"m": "male", "f": "female", "male": "male", "female": "female"}))
    df["marital_status"] = df["marital_status"].str.strip().str.lower()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """The six engineered features from notebook section B3."""
    df = df.copy()
    df["policy_start_date"] = pd.to_datetime(df["policy_start_date"])
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["days_since_policy_start"] = (df["claim_date"] - df["policy_start_date"]).dt.days
    df["total_hospital_days"] = df["icu_days"] + df["length_of_stay"]
    df["icu_ratio"] = df["icu_days"] / (df["length_of_stay"] + 1)
    df["comorbidity_count"] = (df["has_diabetes"].fillna(0) + df["has_hypertension"].fillna(0)
                               + df["family_history_cardiac"])
    df["bmi_risk"] = (df["bmi"] >= 30).astype(float)
    df["claim_month"] = df["claim_date"].dt.month
    return df


def make_preprocessor():
    """Fresh, unfitted preprocessor (impute -> scale numerics, impute -> one-hot categoricals)."""
    plain = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    flags = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0)),
                      ("scale", StandardScaler())])
    cats = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                     ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", plain, PLAIN_NUM), ("flag", flags, FLAG_COLS),
                              ("cat", cats, CATEGORICAL_COLS)])
