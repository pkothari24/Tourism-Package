# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Initialize the Hugging Face API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# --- 1. DYNAMIC DATA LOADING ---
# Tries multiple possible path configurations on the GitHub Action Runner
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
    print("❌ CRITICAL ERROR: Could not find 'tourism.csv' anywhere in the workspace.")
    print("Root files available:", os.listdir(current_workspace))
    raise FileNotFoundError("The target tourism.csv file was missing from this runner context.")

# --- 2. LOG AVAILABLE COLUMNS ---
# This prints the columns directly to your GitHub workflow output window
print("\n--- DATA DIAGNOSTICS ---")
print("Available columns in dataset:", df.columns.tolist())
print("------------------------\n")

# --- 3. SAFE UNIQUE ID DROPPING ---
# Automatically searches for variations of UDI column
udi_col = None
for col in df.columns:
    if col.strip().lower() == 'udi':
        udi_col = col
        break

if udi_col:
    df.drop(columns=[udi_col], inplace=True)
    print(f"Successfully dropped unique identifier column: '{udi_col}'")
else:
    print("No unique identifier ('UDI') column found. Skipping drop step safely.")

# --- 4. SAFE CATEGORICAL LABEL ENCODING ---
# Automatically searches for variations of 'Type' column (case-insensitive)
type_col = None
for col in df.columns:
    if col.strip().lower() == 'type':
        type_col = col
        break

if type_col:
    label_encoder = LabelEncoder()
    df[type_col] = label_encoder.fit_transform(df[type_col])
    print(f"Successfully encoded categorical column: '{type_col}'")
else:
    print("Warning: No 'Type' column discovered in dataset rows. Skipping label encoding.")

# --- 5. TARGET SEPARATION AND SPLITTING ---
target_col = 'Failure'

if target_col not in df.columns:
    # Try case-insensitive fallback for 'Failure' target column
    for col in df.columns:
        if col.strip().lower() == 'failure':
            target_col = col
            break

print(f"Using target column: '{target_col}'")
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save processed datasets locally onto the workspace runner
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
        path_in_repo=file_path.split("/")[-1],  # keeps just the raw filename
        repo_id="pkothari24/tourism_project",
        repo_type="dataset",
    )

print("✅ Pipeline execution and Hugging Face hub transfer finished successfully!")
