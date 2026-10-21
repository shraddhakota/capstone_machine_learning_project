# 23CSE301 Machine Learning Capstone

End-to-end ML pipeline covering three tracks: **Regression**, **Classification** and **Clustering**,
with an interactive **Streamlit GUI**.

**Live demo:** https://YOUR-APP-NAME.streamlit.app


## Problem statement and datasets

| Track | Dataset | Problem | Size |
|---|---|---|---|
| Regression | `indian_health_insurance_claims_dataset.csv` | Predict `total_claim_amount` (₹) of a health-insurance claim from patient, policy and hospital features | 8,000 rows, 38 columns |
| Classification | `mental_health_burnout_prediction_dataset.csv` | Predict `Mental_Health_Status` (Healthy / Needs Attention / Critical) | 50,000 rows |
| Clustering | `healthcare_fraud_detection.csv` | Group healthcare claims into behavioural segments without labels | 18,764 claims |

Key design decisions
- **Regression:** itemised billing columns (hospital bill, pre/post-hospitalisation, deductible, co-pay, non-payable items) are excluded because they arithmetically define the target (data leakage). Six engineered features are added (e.g. `total_hospital_days`, `icu_ratio`).
- **Classification:** `Burnout_Score`, `Burnout_Risk` and `AI_Wellness_Recommendation` are dropped as they leak the target.
- **Clustering:** labels (`Is_Fraud`) are used only to interpret clusters after fitting, never for training.
- All preprocessing is fitted on the training split only; `random_state=42`; 80:20 split.

## Results summary

**Regression** (all 10 algorithms on the same test set, ranked by R²)

| Rank | Algorithm | R² | RMSE | MAE |
|---|---|---|---|---|
| 1 | Gradient Boosting (best; tuned R² 0.7651, 5-fold CV R² 0.7817) | 0.7653 | 27,471 | 18,624 |
| 2 | Lasso (tuned; 5-fold CV R² 0.7570) | 0.7459 | 28,583 | 19,889 |
| 3 | Ridge | 0.7459 | 28,584 | 19,889 |
| 4 | Linear Regression | 0.7459 | 28,584 | 19,890 |
| 5 | SVR | 0.7394 | 28,948 | 19,908 |
| 6 | Random Forest | 0.7315 | 29,380 | 20,328 |
| 7 | Polynomial (deg 2) | 0.7164 | 30,194 | 21,413 |
| 8 | Decision Tree | 0.6786 | 32,145 | 22,057 |
| 9 | ElasticNet | 0.6508 | 33,509 | 24,213 |
| 10 | KNN | 0.5422 | 38,364 | 28,700 |

**Classification** (all 10 algorithms, weighted metrics)

| Rank | Algorithm | Accuracy | Precision | Recall | F1 | ROC-AUC (OvR) |
|---|---|---|---|---|---|---|
| 1 | Random Forest (tuned, best) | 0.8273 | 0.8405 | 0.8273 | 0.8309 | 0.9518 |
| 2 | Bagging (Decision Tree) | 0.8284 | 0.8360 | 0.8284 | 0.8307 | 0.9533 |
| 3 | Gradient Boosting | 0.8291 | 0.8341 | 0.8291 | 0.8307 | 0.9539 |
| 4 | Logistic Regression | 0.8285 | 0.8302 | 0.8285 | 0.8291 | 0.9527 |
| 5 | Decision Tree | 0.8215 | 0.8253 | 0.8215 | 0.8229 | 0.9461 |
| 6 | SVM | 0.8163 | 0.8346 | 0.8163 | 0.8208 | 0.9454 |
| 7 | MLP | 0.8168 | 0.8196 | 0.8168 | 0.8180 | 0.9464 |
| 8 | AdaBoost | 0.8165 | 0.8123 | 0.8165 | 0.8137 | 0.8253 |
| 9 | Naive Bayes | 0.8061 | 0.8030 | 0.8061 | 0.7988 | 0.9467 |
| 10 | KNN | 0.7210 | 0.7242 | 0.7210 | 0.7145 | 0.8718 |

**Clustering** (k = 5)

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| K-Means (5,000-claim sample) | 0.2329 | 1.2375 | 1,263 |
| Agglomerative, Ward (5,000-claim sample) | 0.1663 | 1.4327 | 938 |
| K-Means (all 18,764 claims) | 0.2329 | 1.2478 | 4,699 |

K-Means is the better-separated model on all three metrics.

## 4. Environment setup

Python 3.12 is recommended (3.11+ required).

```bash
git clone https://github.com/shraddhakota/capstone_machine_learning_project.git
cd capstone_machine_learning_project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. How to run

- **Notebooks:** open the notebooks in `notebooks/` (Jupyter or Colab) and run all cells top to bottom. Datasets are in `data/`.
- **Retrain the saved models (optional, about 1 minute):** `python app/train_models.py`
- **GUI:** `streamlit run app/app.py` and open http://localhost:8501

The GUI has four tabs: Regression (claim-amount predictor), Classification (mental-health status),
Clustering (claim segment plus PCA plot) and Results (metrics and full comparison tables).
It is deployed publicly on Streamlit Community Cloud (link above).

## 6. AI assistance disclosure

AI usage disclosure- Section 7.5 of the guidelines permits usage of AI for code scaffolding, therefore we have used Claude AI's assistance to debug some of the codes. However, the analyses and interpretations from the data as well as the final insights are completely done by us.


Generative AI (Claude, by Anthropic) was used for code scaffolding of the Streamlit GUI, the model-training
script and the deployment files. Analysis, interpretation and feature-engineering decisions are the team's own.
