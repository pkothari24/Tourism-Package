import os
import pandas as pd
import numpy as np
import joblib

# Sklearn & XGBoost components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score

# Hugging Face Hub components
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow for Experimentation Tracking & Registration
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def main():
    # 1. Setup MLflow Tracking 
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Wellness_Tourism_Package_Prediction")

    # 2. Dynamic Data Loading and Splitting
    current_workspace = os.getcwd()
    raw_data_path = os.path.join(current_workspace, "project", "data", "tourism.csv")
    
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Could not locate base raw dataset at {raw_data_path}")

    print(f"🚀 Found raw data at: {raw_data_path}. Splitting into train/test states dynamically...")
    df = pd.read_csv(raw_data_path)

    # Separate target variable 'ProdTaken' from features
    target_col = 'ProdTaken'
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in the dataset features.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col].squeeze()

    # Perform a clean 80/20 stratification split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Drop CustomerID if present
    if 'CustomerID' in Xtrain.columns:
        Xtrain = Xtrain.drop(columns=['CustomerID'])
        Xtest = Xtest.drop(columns=['CustomerID'])

    # 3. Define Feature Types
    numeric_features = [
        'Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 
        'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 
        'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
    ]
    # Filter to only include features actually present in the CSV to prevent KeyError
    numeric_features = [f for f in numeric_features if f in Xtrain.columns]

    categorical_features = [
        'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched'
    ]
    categorical_features = [f for f in categorical_features if f in Xtrain.columns]

    class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore'), categorical_features)
    )

    xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric='logloss')

    param_grid = {
        'xgbclassifier__n_estimators': [50, 100],
        'xgbclassifier__max_depth': [3, 4],
        'xgbclassifier__learning_rate': [0.05, 0.1],
        'xgbclassifier__colsample_bytree': [0.7, 0.9],
        'xgbclassifier__reg_lambda': [1.0, 1.5]
    }
    model_pipeline = make_pipeline(preprocessor, xgb_model)

    # 4. Start MLflow Tuning Run
    with mlflow.start_run(run_name="XGBoost_Wellness_Package_Tuning") as parent_run:
        train_dataset = mlflow.data.from_pandas(Xtrain, name="travel_wellness_train")
        mlflow.log_input(train_dataset, context="training")

        print("Starting Hyperparameter Optimization via GridSearchCV...")
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_
        print("Best Hyperparameters Found:\n", grid_search.best_params_)
        for param_name, param_value in grid_search.best_params_.items():
            mlflow.log_param(param_name, param_value)

        y_pred_test = best_model.predict(Xtest)
        mlflow.log_metric("test_f1_score", f1_score(ytest, y_pred_test))
        mlflow.log_metric("test_recall", recall_score(ytest, y_pred_test))
        mlflow.log_metric("test_accuracy", accuracy_score(ytest, y_pred_test))

        signature = infer_signature(Xtest, y_pred_test)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="wellness_package_xgb_pipeline",
            signature=signature,
            registered_model_name="Wellness_Tourism_Package_Prediction_Model",
            serialization_format="pickle"
        )

    # 5. Serialize Local Artifact File
    model_filename = "best_wellness_package_model_v1.joblib"
    joblib.dump(best_model, model_filename)
    print(f"✅ Local model artifact saved as {model_filename}")

    # 6. Push Model File to Hugging Face Model Hub
    repo_id = "pkothari24/Tourism-Package"
    api = HfApi(token=os.getenv("HF_TOKEN"))

    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        print(f"Creating model repository '{repo_id}' on Hugging Face...")
        api.create_repo(repo_id=repo_id, repo_type="model", private=False)

    print("Uploading trained model file to Hugging Face Hub...")
    api.upload_file(
        path_or_fileobj=model_filename,
        path_in_repo=model_filename,
        repo_id=repo_id,
        repo_type="model"
    )
    print("🎉 Model successfully deployed to Hugging Face Registry!")

if __name__ == "__main__":
    main()
