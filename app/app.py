"""23CSE301 ML Capstone - interactive GUI (Streamlit), covering all three tracks.

Tab 1  Regression      : predict an insurance claim amount (Gradient Boosting)
Tab 2  Classification  : predict mental-health status (Random Forest)
Tab 3  Clustering      : find a healthcare claim's segment (K-Means, k=5)
Tab 4  Results         : metrics + the notebooks' full model-comparison tables

Run locally:  streamlit run app/app.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
import regression_utils as ru  # noqa: E402
from preprocessing import LABELS  # noqa: E402  (also needed so joblib can unpickle the pipeline)

MODELS = APP_DIR.parent / "models"
st.set_page_config(page_title="ML Capstone Demo", page_icon="🤖", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(MODELS / name)


@st.cache_data
def load_json(name):
    return json.loads((MODELS / name).read_text())


@st.cache_data
def load_csv(name, **kw):
    p = MODELS / name
    return pd.read_csv(p, **kw) if p.exists() else None


def nice(x):
    return str(x).replace("_", " ").title()


reg_schema = load_json("regression_schema.json")
clf_schema = load_json("schema.json")
metrics = load_json("metrics.json")
cprof = load_json("cluster_profiles.json")

st.title("23CSE301 Machine Learning Capstone")
st.caption("Interactive demo of the three trained tracks: regression (insurance claims), "
           "classification (mental health) and clustering (healthcare claims).")
tab_reg, tab_clf, tab_clu, tab_res = st.tabs(
    ["💰 Regression", "🧠 Classification", "🧾 Clustering", "📊 Results"])

# ====================================================================== Tab 1: regression
with tab_reg:
    st.subheader("Predict the total claim amount (₹)")
    st.write("Fill in the patient, policy and hospital details. The Gradient Boosting model "
             "predicts the claim amount **without** using the itemised billing columns "
             "(those define the target, so the notebook excludes them to avoid leakage).")
    num, cat = reg_schema["num"], reg_schema["cat"]
    flag_opts = {"No": 0.0, "Yes": 1.0, "Not recorded": np.nan}

    with st.expander("👤 Patient", expanded=True):
        a, b, c = st.columns(3)
        age = a.slider("Age", int(num["age"]["min"]), int(num["age"]["max"]), int(num["age"]["default"]))
        gender = b.selectbox("Gender", cat["gender"], format_func=nice,
                             index=cat["gender"].index(reg_schema["cat_default"]["gender"]))
        marital = c.selectbox("Marital status", cat["marital_status"], format_func=nice)
        state = a.selectbox("State", cat["state"], format_func=nice,
                            index=cat["state"].index(reg_schema["cat_default"]["state"]))
        city = b.selectbox("City", reg_schema["state_city"][state])
        occupation = c.selectbox("Occupation", cat["occupation_type"], format_func=nice)
        income = a.number_input("Annual income (₹)", 0.0, 50_000_000.0,
                                float(round(num["annual_income_inr"]["default"], -3)), step=10_000.0)
        bmi = b.slider("BMI", 12.0, 50.0, float(round(num["bmi"]["default"], 1)), step=0.1)
        stress = c.slider("Stress level score", 0.0, 10.0,
                          float(round(num["stress_level_score"]["default"], 1)), step=0.1)
        d, e, f = st.columns(3)
        tobacco = d.selectbox("Tobacco usage", cat["tobacco_usage"], format_func=nice)
        alcohol = e.slider("Alcohol units / week", 0, int(num["alcohol_units_per_week"]["max"]),
                           int(num["alcohol_units_per_week"]["default"]))
        activity = f.selectbox("Physical activity", cat["physical_activity_level"], format_func=nice)
        diet = d.selectbox("Diet", cat["diet_type"], format_func=nice)
        diabetes = e.selectbox("Diabetes", list(flag_opts))
        hypertension = f.selectbox("Hypertension", list(flag_opts))
        family_cardiac = d.selectbox("Family history of cardiac disease", ["No", "Yes"])

    with st.expander("📄 Policy & hospital", expanded=True):
        a, b, c = st.columns(3)
        policy_type = a.selectbox("Policy type", cat["policy_type"], format_func=nice)
        sum_insured = b.selectbox("Sum insured (₹)", reg_schema["sum_insured"], index=2,
                                  format_func=lambda v: f"{v:,}")
        agent = c.selectbox("Sales channel", cat["agent_channel"], format_func=nice)
        tier = a.selectbox("Hospital tier", cat["hospital_tier"], format_func=nice)
        room = b.selectbox("Room category", cat["room_category"], format_func=nice)
        cashless = c.selectbox("Cashless claim", cat["cashless_claim"], format_func=nice)
        ps_lo, ps_hi = (dt.date.fromisoformat(x) for x in reg_schema["policy_start"])
        cl_lo, cl_hi, cl_mid = (dt.date.fromisoformat(x) for x in reg_schema["claim_date"])
        pol_start = a.date_input("Policy start date", ps_lo + (ps_hi - ps_lo) / 2, ps_lo, ps_hi)
        claim_date = b.date_input("Claim date", cl_mid, cl_lo, cl_hi)

    with st.expander("🏥 Stay", expanded=True):
        a, b = st.columns(2)
        icu = a.slider("ICU days", 0, int(num["icu_days"]["max"]), int(num["icu_days"]["default"]))
        los = b.slider("Length of stay (days)", 0, int(num["length_of_stay"]["max"]),
                       int(num["length_of_stay"]["default"]))

    if st.button("Predict claim amount", type="primary", key="reg_go"):
        if claim_date < pol_start:
            st.error("Claim date must be on or after the policy start date.")
        else:
            raw = pd.DataFrame([{
                "age": age, "gender": gender, "marital_status": marital, "state": state, "city": city,
                "occupation_type": occupation, "annual_income_inr": income, "bmi": bmi,
                "tobacco_usage": tobacco, "alcohol_units_per_week": alcohol,
                "physical_activity_level": activity, "diet_type": diet,
                "has_diabetes": flag_opts[diabetes], "has_hypertension": flag_opts[hypertension],
                "family_history_cardiac": 1 if family_cardiac == "Yes" else 0,
                "stress_level_score": stress, "policy_type": policy_type, "sum_insured": sum_insured,
                "policy_start_date": pol_start, "claim_date": claim_date, "hospital_tier": tier,
                "room_category": room, "icu_days": icu, "length_of_stay": los,
                "cashless_claim": cashless, "agent_channel": agent}])
            feats = ru.add_features(raw)
            pred = float(load_model("claims_gbr_pipeline.joblib").predict(feats[ru.FEATURE_COLS])[0])
            mae = metrics["regression"]["mae"]
            st.success(f"Predicted claim amount: **₹{pred:,.0f}**")
            st.caption(f"Typical error on unseen claims: about ±₹{mae:,.0f} (test MAE). "
                       f"Predictions are capped at the training 99th percentile (₹{metrics['regression']['cap_value']:,.0f}).")
            st.write("Engineered features computed from your inputs:")
            st.dataframe(feats[["days_since_policy_start", "total_hospital_days", "icu_ratio",
                                "comorbidity_count", "bmi_risk", "claim_month"]].round(2),
                         hide_index=True)

# ================================================================= Tab 2: classification
KEY_INPUTS = ["Anxiety_Score", "Depression_Score", "Mood_Score", "Emotional_Stability",
              "Productivity_Score", "Stress_Level", "Chronic_Stress", "Work_Hours_Per_Week",
              "Sleep_Hours", "Sleep_Quality"]


def make_widget(col, spec, key):
    if spec["type"] == "cat":
        opts = spec["options"]
        return st.selectbox(nice(col), opts, index=opts.index(spec["default"]), key=key)
    lo, hi, d = spec["min"], spec["max"], spec["default"]
    if float(lo).is_integer() and float(hi).is_integer() and float(d).is_integer():
        return st.slider(nice(col), int(lo), int(hi), int(d), key=key)
    return st.slider(nice(col), float(lo), float(hi), float(round(d, 1)), step=0.1, key=key)


with tab_clf:
    st.subheader("Predict mental-health status")
    st.write("Adjust the key indicators (the strongest predictors in the Random Forest). Every other "
             "field defaults to the typical respondent and can be changed in the expander.")
    values = {}
    left, right = st.columns(2)
    for i, col in enumerate(KEY_INPUTS):
        with (left if i % 2 == 0 else right):
            values[col] = make_widget(col, clf_schema[col], key=f"clf_{col}")
    with st.expander("More inputs (demographics, lifestyle, work)"):
        others = [c for c in clf_schema if c not in KEY_INPUTS]
        cols3 = st.columns(3)
        for i, col in enumerate(others):
            with cols3[i % 3]:
                values[col] = make_widget(col, clf_schema[col], key=f"clf_{col}")

    if st.button("Predict status", type="primary", key="clf_go"):
        row = pd.DataFrame([values])[list(clf_schema.keys())]
        proba = load_model("burnout_rf_pipeline.joblib").predict_proba(row)[0]
        pred = LABELS[int(np.argmax(proba))]
        {"Healthy": st.success, "Needs Attention": st.warning, "Critical": st.error}[pred](
            f"Predicted status: **{pred}** (confidence {proba.max():.0%})")
        st.bar_chart(pd.DataFrame({"Probability": proba}, index=LABELS))
        st.caption("Educational demo on a survey dataset. Not a clinical or diagnostic tool.")

# ====================================================================== Tab 3: clustering
with tab_clu:
    st.subheader("Which claim segment does this claim fall into?")
    st.write("K-Means (k = 5) was fitted **without labels** on four behavioural features. Enter a claim "
             "to see its segment and how often that segment's claims were later flagged as fraud "
             "(a post-hoc check, never used for fitting).")
    r = cprof["ranges"]
    c1, c2 = st.columns(2)
    claim = c1.number_input("Claim amount", 1.0, float(r["Claim_Amount"][1]),
                            float(round(r["Claim_Amount"][2])), step=10.0)
    approved = c1.number_input("Approved amount", 0.0, float(r["Approved_Amount"][1]),
                               float(round(0.85 * r["Claim_Amount"][2])), step=10.0)
    delay = c2.slider("Days between service and claim", 0, int(r["Days_Between_Service_and_Claim"][1]),
                      int(r["Days_Between_Service_and_Claim"][2]))
    volume = c2.slider("Claims per provider per month", 0,
                       int(r["Number_of_Claims_Per_Provider_Monthly"][1]),
                       int(r["Number_of_Claims_Per_Provider_Monthly"][2]))

    if st.button("Find segment", type="primary", key="clu_go"):
        art = load_model("kmeans_claims.pkl")
        x = pd.DataFrame([{"Log_Claim_Amount": np.log1p(claim), "Approval_Ratio": approved / claim,
                           "Days_Between_Service_and_Claim": delay,
                           "Number_of_Claims_Per_Provider_Monthly": volume}])[art["features"]]
        xs = art["scaler"].transform(x)
        cl = int(art["kmeans"].predict(xs)[0])
        p = next(q for q in cprof["profiles"] if q["Cluster"] == cl)
        lift = p["fraud_rate"] / cprof["base_fraud_rate"]
        (st.error if lift >= 3 else st.info)(
            f"Segment **{cl}** · {p['claims']:,} historical claims · fraud rate {p['fraud_rate']:.1%} "
            f"({lift:.1f}× the overall {cprof['base_fraud_rate']:.1%})")

        prof = pd.DataFrame(cprof["profiles"]).set_index("Cluster")
        prof.columns = ["Claims", "Median claim", "Median approval ratio", "Median delay (days)",
                        "Median provider claims / month", "Fraud rate"]
        st.dataframe(prof.style.format({"Fraud rate": "{:.1%}", "Median claim": "{:,.0f}",
                                        "Median approval ratio": "{:.2f}"})
                     .apply(lambda s: ["background-color: #ffe08a" if s.name == cl else "" for _ in s],
                            axis=1))

        sample = load_csv("cluster_pca_sample.csv")
        pt = art["pca"].transform(xs)[0]
        var = cprof["pca_variance"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        cmap = plt.get_cmap("tab10")
        for k in sorted(sample["Cluster"].unique()):
            s = sample[sample["Cluster"] == k]
            ax.scatter(s["PC1"], s["PC2"], s=8, alpha=0.5, color=cmap(k), label=f"Cluster {k}")
        ax.scatter(*pt, s=220, marker="*", color="black", edgecolor="white", linewidth=1.2,
                   label="Your claim", zorder=5)
        ax.set_title("K-Means clusters (PCA projection, 3,000-claim sample)")
        ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
        ax.legend(markerscale=2, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Cluster numbers are arbitrary K-Means labels; the highlighted row is this claim's segment.")

# ======================================================================== Tab 4: results
with tab_res:
    m = metrics["regression"]
    st.subheader("Regression — insurance claim amount")
    st.write(f"**Deployed model:** {m['model']} · {m['n_train']:,} train / {m['n_test']:,} test rows · seed 42")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{m['r2']:.4f}")
    c2.metric("RMSE", f"₹{m['rmse']:,.0f}")
    c3.metric("MAE", f"₹{m['mae']:,.0f}")
    t = load_csv("results_regression.csv", index_col=0)
    if t is not None:
        st.write("All 10 algorithms (from `notebooks/regression.ipynb`, ranked by R²)")
        st.dataframe(t)
    st.write("Top feature importances")
    st.bar_chart(pd.Series(m["top_features"], name="Importance"))

    st.divider()
    m = metrics["classification"]
    st.subheader("Classification — mental-health status")
    st.write(f"**Deployed model:** {m['model']} · {m['n_train']:,} train / {m['n_test']:,} test rows · seed 42")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision (wtd)", f"{m['precision_weighted']:.3f}")
    c3.metric("Recall (wtd)", f"{m['recall_weighted']:.3f}")
    c4.metric("F1 (wtd)", f"{m['f1_weighted']:.3f}")
    c5.metric("ROC-AUC (OvR)", f"{m['roc_auc_ovr_weighted']:.3f}")
    st.write("Confusion matrix (test set)")
    st.dataframe(pd.DataFrame(m["confusion_matrix"], index=[f"True: {l}" for l in LABELS],
                              columns=[f"Pred: {l}" for l in LABELS]))
    t = load_csv("results_classification.csv", index_col=0)
    if t is not None:
        st.write("All 10 algorithms (from `notebooks/classification.ipynb`, ranked by weighted F1)")
        st.dataframe(t)
    st.write("Top feature importances")
    st.bar_chart(pd.Series(m["top_features"], name="Importance"))

    st.divider()
    m = metrics["clustering"]
    st.subheader("Clustering — healthcare claims")
    st.write(f"**Deployed model:** {m['model']} on {m['n_claims']:,} claims")
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette (↑)", f"{m['silhouette']:.3f}")
    c2.metric("Davies-Bouldin (↓)", f"{m['davies_bouldin']:.3f}")
    c3.metric("Calinski-Harabasz (↑)", f"{m['calinski_harabasz']:,.0f}")
    t = load_csv("results_clustering.csv", index_col=0)
    if t is not None:
        st.write("K-Means vs Agglomerative (from `notebooks/clustering.ipynb`)")
        st.dataframe(t)
    st.caption("The deployed models are refitted by `app/train_models.py` with the notebooks' settings, "
               "so the last decimal can differ slightly from the notebook tables.")
