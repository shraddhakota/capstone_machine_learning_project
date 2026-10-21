"""Train and save every model used by the Streamlit app (all 3 capstone tracks).

Run from the repo root:  python app/train_models.py
Reproduces the notebooks: same seed (42), same split, same preprocessing, same tuned params.
Also copies the notebooks' final comparison tables into models/ so the app can display them.
"""
import json
import sys
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (accuracy_score, calinski_harabasz_score, confusion_matrix,
                             davies_bouldin_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, str(Path(__file__).parent))
import regression_utils as ru  # noqa: E402
from preprocessing import LABELS, LEAKAGE_COLS, TARGET, BurnoutPreprocessor  # noqa: E402

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, NOTEBOOKS = ROOT / "data", ROOT / "models", ROOT / "notebooks"
MODELS.mkdir(exist_ok=True)
metrics = {}

# =========================================================== 1. REGRESSION (insurance claims)
raw = pd.read_csv(DATA / "indian_health_insurance_claims_dataset.csv")
df = ru.add_features(ru.clean_dataframe(raw))
X, y = df[ru.FEATURE_COLS], df[ru.TARGET]
y_bins = pd.qcut(y, q=10, labels=False, duplicates="drop")      # stratify on target deciles
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y_bins)
cap = y_tr.quantile(0.99)                                       # winsorise at train 99th pct
y_tr, y_te = y_tr.clip(upper=cap), y_te.clip(upper=cap)

reg = Pipeline([("preprocess", ru.make_preprocessor()),
                ("model", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                    max_depth=3, random_state=SEED))]).fit(X_tr, y_tr)
p = reg.predict(X_te)
metrics["regression"] = {
    "model": "Gradient Boosting Regressor (n_estimators=200, learning_rate=0.05, max_depth=3)",
    "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    "r2": r2_score(y_te, p), "rmse": float(np.sqrt(mean_squared_error(y_te, p))),
    "mae": mean_absolute_error(y_te, p), "cap_value": float(cap)}
names = reg.named_steps["preprocess"].get_feature_names_out()
imp = pd.Series(reg.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
metrics["regression"]["top_features"] = imp.head(10).round(4).to_dict()
joblib.dump(reg, MODELS / "claims_gbr_pipeline.joblib", compress=3)

reg_schema = {
    "num": {c: {"min": float(df[c].min()), "max": float(df[c].max()), "default": float(df[c].median())}
            for c in ["age", "annual_income_inr", "bmi", "alcohol_units_per_week",
                      "stress_level_score", "icu_days", "length_of_stay"]},
    "cat": {c: sorted(df[c].dropna().unique().tolist()) for c in
            ["gender", "marital_status", "occupation_type", "tobacco_usage",
             "physical_activity_level", "diet_type", "policy_type", "hospital_tier",
             "room_category", "cashless_claim", "agent_channel", "state"]},
    "cat_default": {c: df[c].mode()[0] for c in ["gender", "marital_status", "occupation_type",
                    "tobacco_usage", "physical_activity_level", "diet_type", "policy_type",
                    "hospital_tier", "room_category", "cashless_claim", "agent_channel", "state"]},
    "state_city": {s: sorted(g["city"].dropna().unique().tolist()) for s, g in df.groupby("state")},
    "sum_insured": sorted(int(v) for v in df["sum_insured"].unique()),
    "policy_start": [str(df["policy_start_date"].min().date()), str(df["policy_start_date"].max().date())],
    "claim_date": [str(df["claim_date"].min().date()), str(df["claim_date"].max().date()),
                   str(df["claim_date"].median().date())],
}
(MODELS / "regression_schema.json").write_text(json.dumps(reg_schema, indent=2))

# =========================================================== 2. CLASSIFICATION (burnout)
bd = pd.read_csv(DATA / "mental_health_burnout_prediction_dataset.csv")
bd = bd.drop_duplicates(subset=[c for c in bd.columns if c != "Person_ID"]).reset_index(drop=True)
Xc = bd.drop(columns=LEAKAGE_COLS + [TARGET])
yc = bd[TARGET].map({l: i for i, l in enumerate(LABELS)})
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.20, random_state=SEED, stratify=yc)
clf = Pipeline([("prep", BurnoutPreprocessor()),
                ("model", RandomForestClassifier(n_estimators=300, max_depth=8,
                                                 random_state=SEED, n_jobs=-1))]).fit(Xc_tr, yc_tr)
pred, proba = clf.predict(Xc_te), clf.predict_proba(Xc_te)
imp = pd.Series(clf.named_steps["model"].feature_importances_,
                index=clf.named_steps["prep"].feature_names_).sort_values(ascending=False)
metrics["classification"] = {
    "model": "Random Forest (n_estimators=300, max_depth=8)",
    "n_train": int(len(Xc_tr)), "n_test": int(len(Xc_te)),
    "accuracy": accuracy_score(yc_te, pred),
    "precision_weighted": precision_score(yc_te, pred, average="weighted"),
    "recall_weighted": recall_score(yc_te, pred, average="weighted"),
    "f1_weighted": f1_score(yc_te, pred, average="weighted"),
    "roc_auc_ovr_weighted": roc_auc_score(label_binarize(yc_te, classes=[0, 1, 2]), proba,
                                          average="weighted", multi_class="ovr"),
    "confusion_matrix": confusion_matrix(yc_te, pred).tolist(),
    "top_features": imp.head(10).round(4).to_dict()}
joblib.dump(clf, MODELS / "burnout_rf_pipeline.joblib", compress=3)

clf_schema = {}
for c in Xc.columns:
    if pd.api.types.is_numeric_dtype(Xc[c]):
        clf_schema[c] = {"type": "num", "min": float(Xc[c].min()), "max": float(Xc[c].max()),
                         "default": float(Xc[c].median())}
    else:
        vc = Xc[c].value_counts()
        clf_schema[c] = {"type": "cat", "options": sorted(vc.index.tolist()), "default": vc.index[0]}
(MODELS / "schema.json").write_text(json.dumps(clf_schema, indent=2))

# =========================================================== 3. CLUSTERING (fraud claims)
fr = pd.read_csv(DATA / "healthcare_fraud_detection.csv").drop(columns=["Provider_ID", "Claim_ID"])
fr = fr.drop_duplicates().reset_index(drop=True)
fr["Log_Claim_Amount"] = np.log1p(fr["Claim_Amount"])
fr["Approval_Ratio"] = fr["Approved_Amount"] / fr["Claim_Amount"]
FEATURES = ["Log_Claim_Amount", "Approval_Ratio", "Days_Between_Service_and_Claim",
            "Number_of_Claims_Per_Provider_Monthly"]
K = 5
scaler = StandardScaler().fit(fr[FEATURES])
Xs = scaler.transform(fr[FEATURES])
km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(Xs)
fr["Cluster"] = km.labels_
pca = PCA(n_components=2, random_state=SEED).fit(Xs)
joblib.dump({"scaler": scaler, "kmeans": km, "pca": pca, "features": FEATURES},
            MODELS / "kmeans_claims.pkl", compress=3)
sel = np.random.RandomState(SEED).choice(len(Xs), 3000, replace=False)
P = pca.transform(Xs[sel])
pd.DataFrame({"PC1": P[:, 0], "PC2": P[:, 1], "Cluster": km.labels_[sel]}).to_csv(
    MODELS / "cluster_pca_sample.csv", index=False)

prof = fr.groupby("Cluster").agg(
    claims=("Cluster", "size"), median_claim=("Claim_Amount", "median"),
    median_approval_ratio=("Approval_Ratio", "median"),
    median_delay_days=("Days_Between_Service_and_Claim", "median"),
    median_provider_monthly=("Number_of_Claims_Per_Provider_Monthly", "median"),
    fraud_rate=("Is_Fraud", "mean")).round(3)
(MODELS / "cluster_profiles.json").write_text(json.dumps({
    "base_fraud_rate": float(fr["Is_Fraud"].mean()),
    "pca_variance": pca.explained_variance_ratio_.round(3).tolist(),
    "ranges": {f: [float(fr[f].min()), float(fr[f].max()), float(fr[f].median())]
               for f in ["Claim_Amount", "Approved_Amount", "Days_Between_Service_and_Claim",
                         "Number_of_Claims_Per_Provider_Monthly"]},
    "profiles": prof.reset_index().to_dict(orient="records")}, indent=2))
metrics["clustering"] = {
    "model": f"K-Means (k={K})", "n_claims": int(len(fr)),
    "silhouette": silhouette_score(Xs, km.labels_, sample_size=5000, random_state=SEED),
    "davies_bouldin": davies_bouldin_score(Xs, km.labels_),
    "calinski_harabasz": calinski_harabasz_score(Xs, km.labels_)}
(MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))


# =========================================================== 4. notebook comparison tables
def notebook_tables(path):
    """Yield every HTML table found in a notebook's cell outputs as a DataFrame."""
    nb = json.loads(Path(path).read_text())
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            html = "".join(out.get("data", {}).get("text/html", []))
            if "<table" in html:
                try:
                    t = pd.read_html(StringIO(html), index_col=0)[0]
                    if isinstance(t.columns, pd.MultiIndex):    # tables with a named index (e.g. 'Rank')
                        t.columns = [c[0] for c in t.columns]
                    yield t
                except Exception:
                    pass


def save_table(path, wanted_cols, nrows=None, keep="last"):
    hits = [t for t in notebook_tables(path)
            if all(c in map(str, t.columns) for c in wanted_cols) and (nrows is None or len(t) == nrows)]
    return hits[-1 if keep == "last" else 0] if hits else None


try:
    t = save_table(NOTEBOOKS / "regression.ipynb", ["R2", "RMSE", "MAE", "Tuned R2"], 10)
    if t is not None:
        t.to_csv(MODELS / "results_regression.csv")
    t = save_table(NOTEBOOKS / "classification.ipynb", ["Accuracy", "ROC-AUC (OvR, wtd)"], 10)
    if t is not None:
        t.to_csv(MODELS / "results_classification.csv")
    t = save_table(NOTEBOOKS / "clustering.ipynb", ["Silhouette (↑)", "Davies-Bouldin (↓)"], 3)
    if t is not None:
        t.to_csv(MODELS / "results_clustering.csv")
except Exception as e:  # tables are display-only; never block training
    print("Could not export notebook tables:", e)

print(json.dumps({k: {a: b for a, b in v.items() if a not in ("confusion_matrix", "top_features")}
                  for k, v in metrics.items()}, indent=2))
print(sorted(p.name for p in MODELS.iterdir()))
