# Directory Structure Guide

```
HunyuanVideo_MLX/
├── ckpts/                     # Model weights directory
│   ├── hunyuan-video-t2v-720p/
│   │   └── transformers/
│   │       └── mp_rank_00_model_states.pt
│   ├── vae/
│   │   └── 884-16c-hy.pt
│   └── text_encoder/
│       └── llm.pt
│
├── configs/                   # Configuration files
│   └── mmgp_example.json     # MMGP configuration example
│
├── results/                   # Generated videos directory
│
├── venv/                      # Python virtual environment
│
├── sample_video_mps.py        # Main video generation script
├── monitor_resources.py       # System resource monitoring tool
├── check_system.py           # System compatibility checker
├── download_weights.py        # Model weights downloader
├── setup_env.sh              # Environment setup script
├── cleanup.sh                # Cleanup utility script
│
├── requirements_mps.txt       # Python dependencies
├── README.md                 # Main documentation
├── README_MPS.md             # MPS-specific documentation
├── QUICKSTART.md             # Quick start guide
└── DIRECTORY_STRUCTURE.md    # This file

Key Directories:
- ckpts/: Contains all model weights (large files, gitignored)
- configs/: Configuration files for different generation modes
- results/: Output directory for generated videos (gitignored)

Key Files:
- sample_video_mps.py: Main script for video generation
- monitor_resources.py: Real-time system resource monitoring
- check_system.py: Verifies system compatibility
- setup_env.sh: Sets up environment variables and virtual environment
- cleanup.sh: Manages disk space and temporary files

Configuration:
- requirements_mps.txt: Python package dependencies
- configs/mmgp_example.json: MMGP configuration for memory-efficient generation

Documentation:
- README.md: Main project documentation
- QUICKSTART.md: Getting started guide
- README_MPS.md: MPS-specific details
- DIRECTORY_STRUCTURE.md: Project organization guide
