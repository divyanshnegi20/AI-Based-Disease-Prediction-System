import sys, json
from re import search
import joblib
import pandas as pd
from pathlib import Path

from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.base import clone

# -------- CONFIG --------
DATA_DIR   = Path(".")
TRAIN_F    = DATA_DIR / "train_split.csv"
VAL_F      = DATA_DIR / "val_split.csv"
TEST_F     = DATA_DIR / "test_set.csv"
LABEL_COL  = "Predicted Disease"
RNG        = 42

# -------- 1) LOAD SPLITS --------
train_df = pd.read_csv(TRAIN_F)
val_df   = pd.read_csv(VAL_F)
test_df  = pd.read_csv(TEST_F)

# -------- 2) DEFINE COLUMNS (must match your actual headers) --------
base_numeric = ["Heart Rate (bpm)", "SpO2 Level (%)",
        "Systolic Blood Pressure (mmHg)",
        "Diastolic Blood Pressure (mmHg)",
        "Body Temperature (Â°C)"]
engineered   = ["Pulse Pressure", "MAP"]
numeric_cols = [c for c in base_numeric + engineered if c in train_df.columns]

categorical_cols = [c for c in [
        "Fall Detection",
        "Heart Rate Alert",
        "SpO2 Level Alert",
        "Blood Pressure Alert",
        "Temperature Alert",
] if c in train_df.columns]

print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)

# -------- 3) PREPROCESSING --------
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", num_pipe, numeric_cols),
    ("cat", cat_pipe, categorical_cols),
])

#---------Near Preprocess step------------------

def add_bp_features(X):
    df2 = X.copy()
    for c in ["Systolic Blood Pressure (mmHg)", "Diastolic Blood Pressure (mmHg)"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    if "Systolic Blood Pressure (mmHg)"in df2.columns and "Diastolic Blood Pressure (mmHg)" in df2.columns:
        sbp = df2["Systolic Blood Pressure (mmHg)"]
        dbp = df2["Diastolic Blood Pressure (mmHg)"]
        df2["Pulse Pressure"] = sbp - dbp
        df2["MAP"] = dbp + (sbp - dbp) / 3.0
    return df2

feat = FunctionTransformer(add_bp_features, validate=False)

# # -------- 4) DATA SPLITS --------
X_train, y_train = train_df.drop(columns=[LABEL_COL]), train_df[LABEL_COL]
X_val,   y_val   = val_df.drop(columns=[LABEL_COL]),   val_df[LABEL_COL]
X_test,  y_test  = test_df.drop(columns=[LABEL_COL]),  test_df[LABEL_COL]

# -------- 5) BASELINE MODELS --------
log_reg = Pipeline([
    ("pre", preprocess),
    ("clf", LogisticRegression(max_iter=300, multi_class="multinomial", class_weight="balanced")),
])
rf = Pipeline([
    ("pre", preprocess),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=RNG, class_weight="balanced_subsample", n_jobs=-1)),
])

for name, model in [("LogisticRegression", log_reg), ("RandomForest", rf)]:
    print(f"\n=== Baseline: {name} ===")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    print("Validation report:\n", classification_report(y_val, preds, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_val, preds))

# -------- 6) HYPERPARAMETER TUNING (RandomForest) --------
X_train, y_train = train_df.drop(columns=[LABEL_COL]), train_df[LABEL_COL]
X_val,   y_val   = val_df.drop(columns=[LABEL_COL]),   val_df[LABEL_COL]

log_reg = Pipeline([
    ("pre", preprocess),
    ("clf", LogisticRegression(
        max_iter=300,
        multi_class="multinomial",
        class_weight="balanced",
        n_jobs=None
    )),
])

rf = Pipeline([
    ("pre", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=1
    )),
])

for name, model in [("LogisticRegression", log_reg), ("RandomForest", rf)]:
    print(f"\n=== Baseline: {name} ===")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    print("Validation report:\n", classification_report(y_val, preds, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_val, preds))

# --------additional step-------------

# pipe: full pipeline (preprocessing + model) to tune
pipe = Pipeline([
    ("feat", feat),
    ("pre", preprocess),
    ("clf", RandomForestClassifier(
        random_state=42,
        class_weight="balanced_subsample"
    )),
])

# param_dist: hyperparameter search space for the classifier inside the pipeline
param_dist = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [8, 12, None],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2],
    "clf__max_features": ["sqrt", "log2"],
}

# cv: stratified K-fold for classification
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# -------- 7) REFIT ON TRAIN+VAL AND EVALUATE ON TEST --------
tuner = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=10,
    scoring="f1_macro",
    cv=cv,
    n_jobs=1,
    verbose=1,
    random_state=42,
)

print("\n=== Tuning RandomForest (CV) ===")
tuner.fit(X_train, y_train)
print("Best CV macro F1:", tuner.best_score_)
print("Best params:", tuner.best_params_)

best_model = tuner.best_estimator_

final_model = clone(best_model)
X_trval = pd.concat([X_train, X_val], axis=0)
y_trval = pd.concat([y_train, y_val], axis=0)
final_model.fit(X_trval, y_trval)

# -------- 8) SAVE FINAL PIPELINE --------
MODEL_F = DATA_DIR / "disease_pipeline.joblib"
joblib.dump(final_model, MODEL_F)
print("Saved final pipeline to:", MODEL_F)

# -------- 9) DEMO INFERENCE -> prediction_log.csv (append-safe) --------

pipe_loaded = joblib.load(MODEL_F)

# if len(sys.argv) == 2 and sys.argv[1].endswith(".csv"):
#     new_df = pd.read_csv(sys.argv[1])
#     preds = pipe_loaded.predict(new_df)
#     rows_to_log = new_df.copy()
#     rows_to_log["prediction"] = preds
# else:
#     demo = {
#         "Heart Rate (bpm)": 90,
#         "SpO2 Level (%)": 85,
#         "Systolic Blood Pressure (mmHg)": 139,
#         "Diastolic Blood Pressure (mmHg)": 57,
#         "Body Temperature (Â°C)": 37,
#         "Fall Detection": "NO",
#         "Heart Rate Alert": "NORMAL",
#         "SpO2 Level Alert": "ABNORMAL",
#         "Blood Pressure Alert": "NORMAL",
#         "Temperature Alert": "NORMAL",
#     }
#     df_demo = pd.DataFrame([demo])
#     pred = pipe_loaded.predict(df_demo)[0]
#     rows_to_log = df_demo.copy()
#     rows_to_log["prediction"] = pred

# rows_to_log["timestamp"] = datetime.now().isoformat(timespec="seconds")
# rows_to_log["model"] = str(MODEL_F.name)

LOG_CSV = Path("prediction_log.csv")
LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

# header_required = not LOG_CSV.exists()
# rows_to_log.to_csv(LOG_CSV, mode="a", header=header_required, index=False, encoding="utf-8")
# print("Logged", len(rows_to_log), "row(s) to:", LOG_CSV)

# -------- 10) APPEND NEW DEMO (idempotent) --------

LOG_CSV = Path("prediction_log.csv")
assert LOG_CSV.exists(), "Run Step 9 once to create prediction_log.csv"

new_demo = {
    "Heart Rate (bpm)": 104,
    "SpO2 Level (%)": 97,
    "Systolic Blood Pressure (mmHg)": 178,
    "Diastolic Blood Pressure (mmHg)": 97,
    "Body Temperature (Â°C)": 36.9,
    "Fall Detection": "NO",
    "Heart Rate Alert": "ABNORMAL",
    "SpO2 Level Alert": "NORMAL",
    "Blood Pressure Alert": "ABNORMAL",
    "Temperature Alert": "NORMAL",
}

pipe_loaded = joblib.load(MODEL_F)
df_new = pd.DataFrame([new_demo])
y_new = pipe_loaded.predict(df_new)[0]

row = df_new.copy()
row["prediction"] = y_new
row["timestamp"] = datetime.now().isoformat(timespec="seconds")
row["model"] = str(MODEL_F.name)

row.to_csv(LOG_CSV, mode="a", header=False, index=False, encoding="utf-8")
print("Appended 1 new row to:", LOG_CSV)