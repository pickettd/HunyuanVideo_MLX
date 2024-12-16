import os
import requests
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def main():
    # Create directories
    base_dir = Path("ckpts")
    model_dir = base_dir / "hunyuan-video-t2v-720p" / "transformers"
    vae_dir = base_dir / "vae"
    text_encoder_dir = base_dir / "text_encoder"
    
    for dir_path in [model_dir, vae_dir, text_encoder_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Download model weights from Hugging Face
    print("Downloading model weights...")
    try:
        # Main model
        hf_hub_download(
            repo_id="tencent/HunyuanVideo",
            filename="hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            local_dir=base_dir,
            local_dir_use_symlinks=False
        )
        
        # VAE model
        hf_hub_download(
            repo_id="tencent/HunyuanVideo",
            filename="vae/884-16c-hy.pt",
            local_dir=base_dir,
            local_dir_use_symlinks=False
        )
        
        # Text encoder model
        hf_hub_download(
            repo_id="tencent/HunyuanVideo",
            filename="text_encoder/llm.pt",
            local_dir=base_dir,
            local_dir_use_symlinks=False
        )
        
        print("\nModel weights downloaded successfully!")
        print("\nDirectory structure:")
        print("ckpts/")
        print("├── hunyuan-video-t2v-720p/")
        print("│   └── transformers/")
        print("│       └── mp_rank_00_model_states.pt")
        print("├── vae/")
        print("│   └── 884-16c-hy.pt")
        print("└── text_encoder/")
        print("    └── llm.pt")
        
    except Exception as e:
        print(f"\nError downloading weights: {str(e)}")
        print("\nPlease manually download the weights from https://huggingface.co/tencent/HunyuanVideo")
        print("and place them in the following structure:")
        print("ckpts/")
        print("├── hunyuan-video-t2v-720p/")
        print("│   └── transformers/")
        print("│       └── mp_rank_00_model_states.pt")
        print("├── vae/")
        print("│   └── 884-16c-hy.pt")
        print("└── text_encoder/")
        print("    └── llm.pt")

if __name__ == "__main__":
    main()
