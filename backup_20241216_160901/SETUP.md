# HunyuanVideo Setup Guide

## Prerequisites

1. Hardware Requirements:
- NVIDIA GPU with CUDA support
- Minimum: 45GB GPU memory for 544x960px129f
- Recommended: 60GB GPU memory for 720x1280px129f
- Tested on single 80G GPU

2. Software Requirements:
- Operating System: Linux
- Python 3.10 or later
- CUDA 11.8 or 12.0+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tencent/HunyuanVideo
cd HunyuanVideo
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate HunyuanVideo
```

3. Install dependencies:
```bash
# Install pip dependencies
python -m pip install -r requirements.txt

# Install flash attention v2 for acceleration (requires CUDA 11.8 or above)
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.5.9.post1
```

## Download Models

HunyuanVideo requires three model components:

1. Main HunyuanVideo Model:
```bash
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
```

2. Text Encoder (MLLM):
```bash
# Download llava-llama-3-8b
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./ckpts/llava-llama-3-8b-v1_1-transformers

# Process text encoder
python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py \
    --input_dir ckpts/llava-llama-3-8b-v1_1-transformers \
    --output_dir ckpts/text_encoder
```

3. CLIP Model (Secondary Text Encoder):
```bash
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./ckpts/text_encoder_2
```

## Directory Structure

After setup, your directory should look like this:
```
HunyuanVideo/
  ├──ckpts/
  │  ├──hunyuan-video-t2v-720p/
  │  │  ├──transformers/
  │  │  ├──vae/
  │  ├──text_encoder/
  │  ├──text_encoder_2/
```

## Generate Videos

Basic usage:
```bash
python sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 30 \
    --prompt "a cat is running, realistic." \
    --flow-reverse \
    --save-path ./results
```

## Supported Resolutions

| Resolution | h/w=9:16 | h/w=16:9 | h/w=4:3 | h/w=3:4 | h/w=1:1 |
|------------|----------|----------|---------|---------|---------|
| 540p | 544x960 | 960x544 | 624x832 | 832x624 | 720x720 |
| 720p | 720x1280 | 1280x720 | 1104x832 | 832x1104 | 960x960 |

## Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| --prompt | None | Text prompt for video generation |
| --video-size | 720 1280 | Size of generated video |
| --video-length | 129 | Length of generated video |
| --infer-steps | 30 | Number of sampling steps |
| --embedded-cfg-scale | 6.0 | Embedded Classifier free guidance scale |
| --flow-shift | 9.0 | Shift factor for flow matching schedulers |
| --flow-reverse | False | If reverse learning/sampling from t=1 -> t=0 |
| --neg-prompt | None | Negative prompt for video generation |
| --seed | 0 | Random seed for generation |
| --use-cpu-offload | False | Use CPU offload to save memory |
| --save-path | ./results | Path to save generated video |

## Troubleshooting

1. Memory Issues:
- For high-resolution videos, enable CPU offload with `--use-cpu-offload`
- Try lower resolution settings (540p instead of 720p)
- Monitor GPU memory usage during generation

2. Model Loading Issues:
- Verify all model files are downloaded correctly
- Check file permissions and directory structure
- Run `python check_system.py` to verify setup

3. Generation Quality:
- Adjust `--infer-steps` for quality vs speed tradeoff
- Try different `--embedded-cfg-scale` values
- Use `--flow-reverse` for potentially better results
- Experiment with different prompts and negative prompts
