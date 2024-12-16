import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download, HfApi, login
from loguru import logger

def setup_logging():
    """Configure logging"""
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}")
    logger.add("download_weights.log", rotation="100 MB")

def check_hf_token():
    """Check if HF token is set and valid"""
    token = os.environ.get('HF_TOKEN')
    if not token:
        logger.warning("HF_TOKEN environment variable not set")
        logger.info("Please set your Hugging Face token:")
        logger.info("export HF_TOKEN='your_token_here'")
        logger.info("Or login using: huggingface-cli login")
        return False
    
    try:
        api = HfApi(token=token)
        api.whoami()
        login(token=token)
        return True
    except Exception as e:
        logger.error(f"Error validating token: {str(e)}")
        return False

def create_directories():
    """Create necessary directories"""
    base_dir = Path("ckpts")
    dirs = [
        base_dir / "hunyuan-video-t2v-720p" / "transformers",
        base_dir / "vae",
        base_dir / "text_encoder"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def download_model_files():
    """Download model files from Hugging Face"""
    model_files = {
        "main": {
            "repo_id": "tencent/HunyuanVideo",
            "filename": "hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            "local_path": "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
        },
        "vae": {
            "repo_id": "tencent/HunyuanVideo",
            "filename": "vae/vae.safetensors",
            "local_path": "ckpts/vae/884-16c-hy.pt"
        },
        "text_encoder": {
            "repo_id": "tencent/HunyuanVideo",
            "filename": "text_encoder/text_encoder.safetensors",
            "local_path": "ckpts/text_encoder/llm.pt"
        }
    }

    for model_name, info in model_files.items():
        logger.info(f"\nDownloading {model_name} model...")
        try:
            file_path = hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["filename"],
                local_dir="ckpts",
                local_dir_use_symlinks=False,
                resume_download=True,
                force_download=False
            )
            
            # Move to correct location if needed
            target_path = Path(info["local_path"])
            if Path(file_path) != target_path:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                Path(file_path).rename(target_path)
            
            size_gb = target_path.stat().st_size / (1024**3)
            logger.success(f"✓ Successfully downloaded {model_name} model ({size_gb:.1f} GB)")
            
        except Exception as e:
            logger.error(f"✗ Error downloading {model_name} model: {str(e)}")
            logger.warning(f"Please check the model repository for updates:")
            logger.warning("https://huggingface.co/tencent/HunyuanVideo")

def verify_downloads():
    """Verify downloaded files"""
    logger.info("\nVerifying downloads...")
    
    expected_files = {
        "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt": "Main model",
        "ckpts/vae/884-16c-hy.pt": "VAE model",
        "ckpts/text_encoder/llm.pt": "Text encoder"
    }
    
    all_present = True
    for file_path, desc in expected_files.items():
        path = Path(file_path)
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            logger.success(f"✓ {desc} present ({size_gb:.1f} GB)")
        else:
            logger.error(f"✗ {desc} missing: {file_path}")
            all_present = False
    
    return all_present

def main():
    setup_logging()
    logger.info("Starting HunyuanVideo model weights download")
    
    if not check_hf_token():
        sys.exit(1)
    
    create_directories()
    download_model_files()
    
    if verify_downloads():
        logger.success("\nAll model weights downloaded successfully!")
        logger.info("Next steps:")
        logger.info("1. Run 'python check_system.py' to verify your setup")
        logger.info("2. Follow the examples in QUICKSTART.md to generate videos")
    else:
        logger.warning("\nSome model files are missing.")
        logger.info("Please check the error messages above and try downloading again.")
        logger.info("If issues persist, check the model repository for updates:")
        logger.info("https://huggingface.co/tencent/HunyuanVideo")

if __name__ == "__main__":
    main()
