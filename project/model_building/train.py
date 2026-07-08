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

    # 2. Remote Data Loading Via Raw Endpoints
    repo_id = "pkothari24/Tourism-Package"
    
    Xtrain_path = f"https://huggingface.co/datasets/{repo_id}/resolve/main/Xtrain.csv"
    Xtest_path = f"https://huggingface.co/datasets/{repo_id}/resolve/main/Xtest.csv"
    ytrain_path = f"https://huggingface.co/datasets/{repo_id}/resolve/main/ytrain.csv"
    ytest_path = f"https://huggingface.co/datasets/{repo_id}/resolve/main/ytest.csv"

    print(f"🚀 Fetching dataset splits directly from Hugging Face Repository: {repo_id}")
    
    storage_options = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"} if os.getenv("HF_TOKEN") else None
    
    try:
        Xtrain = pd.read_csv(Xtrain_path, storage_options=storage_options)
        Xtest = pd.read_csv(Xtest_path, storage_options=storage_options)
        ytrain = pd.read_csv(ytrain_path, storage_options=storage_options).squeeze()
        ytest = pd.read_csv(ytest_path, storage_options=storage_options).squeeze()
        print("✅ Success! Splits fetched and loaded into Pandas memory frames.")
    except Exception as e:
        print(f"❌ Failed downloading splits directly from Hugging Face Hub: {e}")
        print("Falling back to local fallback directory scan...")
        current_workspace = os.getcwd()
        possible_dirs = [os.path.join(current_workspace, "project", "model_building"), current_workspace]
        data_dir = None
        for directory in possible_dirs:
            if os.path.exists(os.path.join(directory, "Xtrain.csv")):
                data_dir = directory
                break
        if not data_dir:
            raise FileNotFoundError("Data split structures could not be retrieved remotely or locally.")
        Xtrain = pd.read_csv(os.path.join(data_dir, "Xtrain.csv"))
        Xtest = pd.read_csv(os.path.join(data_dir, "Xtest.csv"))
        ytrain = pd.read_csv(os.path.join(data_dir, "ytrain.csv")).squeeze()
        ytest = pd.read_csv(os.path.join(data_dir, "ytest.csv")).squeeze()

    # Drop CustomerID if it hasn't been dropped in the data preparation phase
    if 'CustomerID' in Xtrain.columns:
        Xtrain = Xtrain.drop(columns=['CustomerID'])
        Xtest = Xtest.drop(columns=['CustomerID'])

    # 3. Define Feature Types based on Travel Dataset Attributes
    numeric_features = [
        'Age', 'CityTier', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 
        'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 
        'MonthlyIncome', 'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
    ]

    categorical_features = [
        'TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'ProductPitched'
    ]

    # Calculate scale_pos_weight to handle class imbalance
    class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

    # Preprocessing pipeline matching travel attributes
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore'), categorical_features)
    )

    # Initialize XGBoost Classifier
    xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric='logloss')

    # Define hyperparameter grid
    param_grid = {
        'xgbclassifier__n_estimators': [50, 100],
        'xgbclassifier__max_depth': [3, 4],
        'xgbclassifier__learning_rate': [0.05, 0.1],
        'xgbclassifier__colsample_bytree': [0.7, 0.9],
        'xgbclassifier__reg_lambda': [1.0, 1.5]
    }

    model_pipeline = make_pipeline(preprocessor, xgb_model)

    # 4. Start MLflow Parent Run for Hyperparameter Tuning
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

        # Model Inference
        y_pred_train = best_model.predict(Xtrain)
        y_pred_test = best_model.predict(Xtest)

        # Calculate metrics
        train_f1 = f1_score(ytrain, y_pred_train)
        test_f1 = f1_score(ytest, y_pred_test)
        test_recall = recall_score(ytest, y_pred_test)
        test_precision = precision_score(ytest, y_pred_test)
        test_accuracy = accuracy_score(ytest, y_pred_test)

        # Log Metrics
        mlflow.log_metric("train_f1_score", train_f1)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_accuracy", test_accuracy)

        print(f"\nTest Set Metrics Logged - F1: {test_f1:.4f}, Recall: {test_recall:.4f}")

        signature = infer_signature(Xtest, y_pred_test)

        model_name = "Wellness_Tourism_Package_Prediction_Model"
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="wellness_package_xgb_pipeline",
            signature=signature,
            registered_model_name=model_name,
            serialization_format="pickle"
        )
        print(f"Model logged in MLflow Registry as '{model_name}'")

    # 5. Serialize Local Artifact File
    model_filename = "best_wellness_package_model_v1.joblib"
    joblib.dump(best_model, model_filename)

    # 6. Push Verified Model Version to your Hugging Face Space
    repo_type = "model"

    api = HfApi(token=os.getenv("HF_TOKEN"))

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Hugging Face Model Hub '{repo_id}' exists.")
    except RepositoryNotFoundError:
        print(f"Hugging Face Model Hub '{repo_id}' not found. Creating a new repository...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model Hub Repo '{repo_id}' created.")

    print(f"Uploading travel model file to Hugging Face Hub...")
    api.upload_file(
        path_or_fileobj=model_filename,
        path_in_repo=model_filename,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Management pipeline complete! Model deployed to HF Registry safely.")

if __name__ == "__main__":
    main()
