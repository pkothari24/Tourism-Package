from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

repo_id = "pkothari24/Tourism-Package"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new repository...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

# --- WORKSPACE-SAFE PATH SEARCH ---
cwd = Path.cwd()
possible_paths = [
    cwd / "project" / "data",
    cwd / "Tourism-Package" / "project" / "data",
    Path(__file__).resolve().parents[1] / "data",
    cwd / "data"
]

target_folder = None
for path in possible_paths:
    print(f"Checking potential path: {path}")
    if path.is_dir():
        target_folder = path
        break

if target_folder is None:
    print(f"❌ Target path matching failed. Visible items in {cwd}:", os.listdir(str(cwd)))
    raise FileNotFoundError(f"Data directory could not be located on this environment.")

print(f"🚀 Match Found! Uploading from: {target_folder}")

api.upload_folder(
    folder_path=str(target_folder),
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Upload completed successfully!")
