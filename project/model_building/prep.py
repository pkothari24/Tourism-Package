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

# Initialize the API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# --- FIXED: Read the file locally from the runner workspace ---
# Since the script runs from the repository root, point to the file directly
DATASET_PATH = os.path.join(os.getcwd(), "project", "model_building", "tourism.csv")

print(f"Loading dataset from: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
    # Fallback in case your folder capitalization varies
    DATASET_PATH = os.path.join(os.getcwd(), "tourism_project", "data", "tourism.csv")

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['UDI'], inplace=True)

# Encoding the categorical 'Type' column
label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])

target_col = 'Failure'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save split outputs locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

print("Uploading processed splits to Hugging Face...")
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="pkothari24/tourism_project",
        repo_type="dataset",
    )

print("All files processed and uploaded successfully!")
