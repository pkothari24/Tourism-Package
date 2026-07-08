from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "pkothari24/Tourism-Package"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if repository exists on Hugging Face
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

# --- UNBREAKABLE PATH FINDER ---
current_workspace = os.getcwd()

# This checks every single possible folder path combinations on the runner machine
possible_paths = [
    os.path.join(current_workspace, "project", "data"),
    os.path.join(current_workspace, "Tourism-Package", "project", "data"),
    os.path.join(current_workspace, "tourism_project", "data"),
    os.path.join(current_workspace, "Tourism-Package", "tourism_project", "data"),
    os.path.join(current_workspace, "data")
]

target_folder = None
for path in possible_paths:
    print(f"Searching for data folder at: {path}")
    if os.path.exists(path) and os.path.isdir(path):
        target_folder = path
        break

# --- EXECUTE UPLOAD ---
if target_folder:
    print(f"🚀 MATCH FOUND! Uploading from: {target_folder}")
    api.upload_folder(
        folder_path=target_folder,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("✅ Upload completed successfully!")
else:
    print("❌ CRITICAL ERROR: Could not find any data directory anywhere.")
    print("Root directory files visible to runner:", os.listdir(current_workspace))
    raise FileNotFoundError("The target data folder structure was not found on this machine.")