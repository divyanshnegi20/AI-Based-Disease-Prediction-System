# AI-Based-Disease-Prediction-System
This AI model is trained on a dataset that contains the test samples of a person of health conditioning and then based on the information it predicts what might the disease would be the person is suffering from like heart disease, asthma, hypertension etc.

At first it reads a dataset in this case it has read Sythetic_patient-HealthCare-Monitoring_dataset.csv
It splits the dataset into two part in 80-20% that is train_set.csv and test_set.csv
The test_set.csv is further spilts into 9:1 that gives us 72% of the dataset that should be untouched and 8% dataset where the model has been firstly tested.
Then it gives us the prediction using any classifier like RandomForestClassifier
Builds a preprocessing + model pipeline with feature engineering.
Trains baselines, tunes Random Forest with CV, refits on train+val, and evaluates on test.
Saves a joblib pipeline for inference.
Supports demo and batch predictions, and appends every prediction to prediction_log.csv with timestamp and model id.
How it works:
Preprocessing ->
  - Numeric: median imputation + StandardScaler.
  - Categorical: most frequent imputation + OneHotEncoder(handle_unknown="ignore").
Feature engineering: FunctionTransformer add_bp_features computes "Pulse Pressure" and "MAP" from systolic/diastolic during both training and inference.
Models->
  - Baselines: Logistic Regression, Random Forest.
  - Tuning: RandomizedSearchCV on Random Forest with StratifiedKFold CV.
Final: best estimator cloned and refit on train+val; evaluate once on test.
Persistence: Final pipeline saved to disease_pipeline.joblib.
