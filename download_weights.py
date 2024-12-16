import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from huggingface_hub import hf_hub_download, list_repo_files

def setup_logging():
    """Configure logging"""
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}")

def create_directories():
    """Create necessary directories"""
    dirs = [
        "ckpts/hunyuan-video-t2v-720p/transformers",
        "ckpts/vae",
        "ckpts/text_encoder",
        "ckpts/text_encoder_2",
        "ckpts/llava-llama-3-8b-v1_1-transformers"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def download_llava_model(token):
    """Download llava model files"""
    logger.info("\nDownloading llava model...")
    repo_id = "xtuner/llava-llama-3-8b-v1_1-transformers"
    
    try:
        # Get list of all files in the repository
        files = list_repo_files(repo_id, token=token)
        logger.info(f"Found {len(files)} files in {repo_id}")
        
        # Download each file
        for file in files:
            if file.endswith('.md') or file.endswith('.git'):
                continue
                
            logger.info(f"Downloading {file}...")
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=file,
                token=token,
                local_dir="ckpts/llava-llama-3-8b-v1_1-transformers"
            )
            logger.success(f"Downloaded {file} to {file_path}")
            
        # Process for text encoder
        logger.info("\nProcessing llava model for text encoder...")
        subprocess.run(
            ["python", "hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py",
             "--input_dir", "ckpts/llava-llama-3-8b-v1_1-transformers",
             "--output_dir", "ckpts/text_encoder"],
            check=True
        )
        logger.success("Successfully processed llava model for text encoder")
        
    except Exception as e:
        logger.error(f"Error with llava model: {str(e)}")
        return False
    
    return True

def download_models():
    """Download required models using huggingface_hub"""
    # Load token from .env
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    
    try:
        # Download main model
        logger.info("Downloading main model...")
        main_model = hf_hub_download(
            repo_id="tencent/HunyuanVideo",
            filename="hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            token=token,
            local_dir="ckpts"
        )
        logger.success(f"Downloaded main model to {main_model}")
        
        # Download VAE model
        logger.info("\nDownloading VAE model...")
        vae_model = hf_hub_download(
            repo_id="tencent/HunyuanVideo",
            filename="hunyuan-video-t2v-720p/vae/pytorch_model.pt",
            token=token,
            local_dir="ckpts"
        )
        logger.success(f"Downloaded VAE model to {vae_model}")
        
        # Download llava model
        if not download_llava_model(token):
            logger.warning("Failed to download or process llava model")
        
        # Download CLIP model files
        logger.info("\nDownloading CLIP model...")
        clip_files = [
            "config.json",
            "model.safetensors",
            "preprocessor_config.json"
        ]
        for filename in clip_files:
            file_path = hf_hub_download(
                repo_id="openai/clip-vit-large-patch14",
                filename=filename,
                token=token,
                local_dir="ckpts/text_encoder_2"
            )
            logger.success(f"Downloaded {filename} to {file_path}")
            
        logger.success("\nAll models downloaded successfully!")
        
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        sys.exit(1)

def main():
    setup_logging()
    logger.info("Starting HunyuanVideo model weights download")
    
    # Check for .env file
    if not Path(".env").exists():
        logger.error(".env file not found")
        logger.info("Please copy .env.example to .env and set your HF_TOKEN")
        sys.exit(1)
    
    create_directories()
    download_models()

if __name__ == "__main__":
    main()
