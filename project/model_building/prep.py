# for data manipulation
import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Initialize the Hugging Face API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Target repository definitions (Matched to your actual HF repo name)
repo_id = "pkothari24/Tourism-Package"
repo_type = "dataset"

# --- 1. ENSURE DATASET REPOSITORY EXISTS ON HUGGING FACE ---
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Hugging Face dataset repo '{repo_id}' verified.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating it now...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Successfully created repository: {repo_id}")

# --- 2. DYNAMIC DATA LOADING ---
current_workspace = os.getcwd()
possible_inputs = [
    os.path.join(current_workspace, "project", "model_building", "tourism.csv"),
    os.path.join(current_workspace, "Tourism-Package", "project", "model_building", "tourism.csv"),
    os.path.join(current_workspace, "tourism_project", "data", "tourism.csv"),
    os.path.join(current_workspace, "project", "data", "tourism.csv")
]

DATASET_PATH = None
for path in possible_inputs:
    if os.path.exists(path):
        DATASET_PATH = path
        break

if DATASET_PATH:
    print(f"🚀 Found dataset at: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")
else:
    raise FileNotFoundError("The target tourism.csv file was missing from this runner context.")

# --- 3. DROP IDENTIFIER COLUMNS ---
cols_to_drop = ['Unnamed: 0', 'CustomerID']
for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"Dropped identifier column: '{col}'")

# --- 4. ENCODE ALL CATEGORICAL COLUMNS ---
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns found for encoding: {categorical_cols}")

label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")
    df[col] = label_encoder.fit_transform(df[col])
    print(f"Successfully encoded: '{col}'")

# --- 5. TARGET SEPARATION AND SPLITTING ---
target_col = 'ProdTaken'

if target_col not in df.columns:
    raise KeyError(f"Expected target column '{target_col}' was not found in the dataset.")

print(f"Using target column: '{target_col}'")
X = df.drop(columns=[target_col])
y = df[target_col]

# Handle remaining numerical missing values
X = X.fillna(X.median(numeric_only=True))

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save processed datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# --- 6. HUGGING FACE TARGET UPLOAD ---
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

print("\nUploading processed training components to Hugging Face...")
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id=repo_id,
        repo_type=repo_type,
    )

print("✅ Pipeline execution and Hugging Face hub transfer finished successfully!")
