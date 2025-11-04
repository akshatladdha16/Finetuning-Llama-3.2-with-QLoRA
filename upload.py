from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv()
api = HfApi(token=os.getenv("HF_API_TOKEN"))
api.upload_folder(
    folder_path="./llama-chemistry-model/checkpoint-200",
    repo_id="akshatladdha16/Llama-3.2-3B-Chemistry-Tutor-LoRA",
    repo_type="model",
)
