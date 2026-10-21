"""Preprocessing for the burnout / mental-health classifier.

Re-implements the notebook's steps (classification.ipynb, sections B1-B3) as one
sklearn-compatible transformer so training and the Streamlit app share exactly the same
code. Everything is learned on the TRAIN split only (no leakage).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Mental_Health_Status"
LABELS = ["Healthy", "Needs Attention", "Critical"]
# Dropped in the notebook: ID + columns that leak the target (Burnout_* and the AI recommendation)
LEAKAGE_COLS = ["Person_ID", "Burnout_Score", "Burnout_Risk", "AI_Wellness_Recommendation"]

ORDINAL_MAPS = {
    "Sleep_Quality": {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3},
    "Stress_Level": {"Low": 0, "Moderate": 1, "High": 2},
    "Exercise_Frequency": {"Never": 0, "Rarely": 1, "Weekly": 2, "Daily": 3},
    "Support_System": {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3},
    "Education_Level": {"High School": 0, "Diploma": 1, "Bachelor": 2, "Master": 3, "PhD": 4},
}
BINARY_COLS = ["Alcohol_Consumption", "Smoking", "Healthy_Diet", "Chronic_Stress",
               "Family_History_Mental_Illness", "Therapy_Attendance"]
NOMINAL_COLS = ["Gender", "Country", "Occupation", "Employment_Status", "Remote_Work"]
CAP_COLS = ["Age", "Monthly_Income_USD", "Work_Hours_Per_Week", "Coffee_Cups_Per_Day"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    # Sleep_Deficit: hours short of an 8h night (chronic short sleep is a burnout driver)
    X["Sleep_Deficit"] = (8 - X["Sleep_Hours"]).clip(lower=0)
    # Digital_Overload_Hours: combined screen + social media + gaming exposure
    X["Digital_Overload_Hours"] = (X["Screen_Time_Hours"].fillna(0)
                                   + X["Social_Media_Hours"].fillna(0)
                                   + X["Gaming_Hours"].fillna(0))
    return X


class BurnoutPreprocessor(BaseEstimator, TransformerMixin):
    """Feature engineering -> impute -> IQR cap -> ordinal/binary/one-hot encode -> scale."""

    def fit(self, X, y=None):
        X = add_engineered_features(X)
        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.num_imp_ = SimpleImputer(strategy="median").fit(X[self.num_cols_])
        self.cat_imp_ = SimpleImputer(strategy="most_frequent").fit(X[self.cat_cols_])
        Xi = self._impute(X)
        self.bounds_ = {}
        for c in CAP_COLS:
            q1, q3 = Xi[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[c] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        Xi = self._cap_and_encode(Xi, fit=True)
        self.scaler_ = StandardScaler().fit(Xi)
        self.feature_names_ = Xi.columns.tolist()
        return self

    def _impute(self, X):
        X = X.copy()
        X[self.num_cols_] = self.num_imp_.transform(X[self.num_cols_])
        X[self.cat_cols_] = self.cat_imp_.transform(X[self.cat_cols_])
        return X

    def _cap_and_encode(self, X, fit=False):
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        for c, m in ORDINAL_MAPS.items():
            X[c] = X[c].map(m)
        for c in BINARY_COLS:
            X[c] = X[c].map({"Yes": 1, "No": 0})
        if fit:
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=int).fit(X[NOMINAL_COLS])
        oh = pd.DataFrame(self.ohe_.transform(X[NOMINAL_COLS]),
                          columns=self.ohe_.get_feature_names_out(NOMINAL_COLS), index=X.index)
        return pd.concat([X.drop(columns=NOMINAL_COLS), oh], axis=1)

    def transform(self, X):
        X = self._impute(add_engineered_features(X))
        X = self._cap_and_encode(X)
        X = X.reindex(columns=self.feature_names_, fill_value=0)
        return self.scaler_.transform(X)
