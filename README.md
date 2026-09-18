AI usage disclosure- Section 7.5 of the guidelines permits usage of AI for code scaffolding, therefore we have used Claude AI's assistance to debug some of the codes. However, the analyses and interpretations from the data as well as the final insights are completely done by us.

Workflow

The notebook follows this pipeline:

Raw Dataset
    ↓
Data Audit
    ↓
Data Cleaning & Repair
    ↓
Exploratory Data Analysis
    ↓
Outlier Analysis / Winsorisation
    ↓
Feature Engineering
    ↓
Data Leakage Control
    ↓
Train/Test Split
    ↓
Preprocessing Pipeline
    ├── Missing-value imputation
    ├── Standard scaling
    └── One-hot encoding
    ↓
Training models
    ↓
Model Comparison
    ↓
Hyperparameter Tuning
    ↓
5-Fold Cross Validation
    ↓
Residual Analysis
    ↓
Feature Importance
    ↓
Final Results

** Technologies Used

Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook



