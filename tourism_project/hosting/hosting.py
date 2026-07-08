import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

def main():
    # 1. Initialize Hugging Face API Client with your write-access token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is missing. Please set it before running this script.")

    api = HfApi(token=hf_token)

    # 2. Configure deployment parameters 
    # Recommended to update the target repo name to match your new Travel use-case
    repo_id = "pkothari24/Tourism-Package"
    repo_type = "space"

    # 3. Ensure the Space repository exists before pushing code
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Hugging Face Space '{repo_id}' exists. Preparing to push updates...")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating a new Streamlit space repository...")
        # SDK spaces default to Streamlit unless specified otherwise
        create_repo(
            repo_id=repo_id, 
            repo_type=repo_type, 
            space_sdk="streamlit",
            private=False
        )
        print(f"Space '{repo_id}' successfully created.")

    # 4. Upload the deployment assets folder containing app.py and requirements.txt
    print(f"Uploading deployment files from 'project/deployment' to Space: {repo_id}...")
    api.upload_folder(
        folder_path="project/deployment",     # Local folder containing app.py and requirements.txt
        repo_id=repo_id,                        # Target space destination
        repo_type=repo_type,                    # Space ecosystem
        path_in_repo="",                        # Deploy directly to the root of the space
    )
    print("Deployment upload sequence complete! Your application is compiling on Hugging Face Spaces.")

if __name__ == "__main__":
    main()
