import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import gradio as gr
import random

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.utils.chunked_generation import generate_video_chunks, clear_memory
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.constants import NEGATIVE_PROMPT

def initialize_model(model_path):
    args = parse_args()
    models_root_path = Path(model_path)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    return hunyuan_video_sampler

def generate_video(
    model,
    prompt,
    resolution,
    video_length,
    seed,
    num_inference_steps,
    guidance_scale,
    flow_shift,
    embedded_guidance_scale
):
    seed = None if seed == -1 else seed
    width, height = resolution.split("x")
    width, height = int(width), int(height)
    negative_prompt = ""  # not applicable in the inference
    
    # Calculate optimal chunk size based on resolution and available memory
    total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3)
    
    # Adjust chunk size based on resolution and RAM
    if total_ram >= 64:
        base_chunk_size = 32
    else:
        base_chunk_size = 16
        
    # Further adjust chunk size based on resolution
    resolution_factor = (width * height) / (1280 * 720)  # Relative to 720p
    chunk_size = max(8, int(base_chunk_size / resolution_factor))
    
    logger.info(f"Using chunk size: {chunk_size} frames (RAM: {total_ram:.1f}GB, Resolution: {width}x{height})")
    
    # Clear memory before generation
    clear_memory()
    
    try:
        # Generate video using chunked approach
        outputs = generate_video_chunks(
            model=model,
            prompt=prompt,
            height=height,
            width=width,
            video_length=video_length,
            chunk_size=chunk_size,
            overlap=4,  # 4 frame overlap for smooth transitions
            seed=seed,
            negative_prompt=negative_prompt,
            infer_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=1,
            flow_shift=flow_shift,
            batch_size=1,
            embedded_guidance_scale=embedded_guidance_scale
        )
        
        samples = outputs['samples']
        sample = samples[0].unsqueeze(0)
        
        save_path = os.path.join(os.getcwd(), "gradio_outputs")
        os.makedirs(save_path, exist_ok=True)
        
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        video_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][0]}_{outputs['prompts'][0][:100].replace('/','')}.mp4"
        save_videos_grid(sample, video_path, fps=24)
        logger.info(f'Sample saved to: {video_path}')
        
        # Clear memory after generation
        clear_memory()
        
        return video_path
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            error_msg = (
                "Out of memory error. Try:\n"
                "1. Using a smaller resolution\n"
                "2. Reducing video length\n"
                "3. Closing other applications\n"
                f"4. Current chunk size: {chunk_size} frames - try reducing video length\n"
                f"5. Current resolution: {width}x{height} - try a smaller size"
            )
            raise gr.Error(error_msg)
        raise e

def create_demo(model_path, save_path):
    model = initialize_model(model_path)
    
    with gr.Blocks() as demo:
        gr.Markdown("# Hunyuan Video Generation")
        gr.Markdown("### Memory-Optimized Generation with Chunked Processing")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="A cat walks on the grass, realistic style.")
                with gr.Row():
                    resolution = gr.Dropdown(
                        choices=[
                            # Memory-optimized resolutions
                            ("544x960 (9:16, 540p)", "544x960"),
                            ("960x544 (16:9, 540p)", "960x544"),
                            ("720x720 (1:1, 540p)", "720x720"),
                            ("832x624 (4:3, 540p)", "832x624"),
                            ("624x832 (3:4, 540p)", "624x832"),
                            # Higher resolutions (64GB+ RAM recommended)
                            ("720x1280 (9:16, 720p)", "720x1280"),
                            ("1280x720 (16:9, 720p)", "1280x720"),
                            ("960x960 (1:1, 720p)", "960x960"),
                            ("1104x832 (4:3, 720p)", "1104x832"),
                            ("832x1104 (3:4, 720p)", "832x1104"),
                        ],
                        value="544x960",
                        label="Resolution (Start with 540p for testing)"
                    )
                    video_length = gr.Dropdown(
                        label="Video Length",
                        choices=[
                            ("2s(65f)", 65),
                            ("5s(129f)", 129),
                        ],
                        value=65,  # Default to shorter length for memory efficiency
                    )
                num_inference_steps = gr.Slider(1, 100, value=25, step=1, label="Number of Inference Steps")
                show_advanced = gr.Checkbox(label="Show Advanced Options", value=False)
                with gr.Row(visible=False) as advanced_row:
                    with gr.Column():
                        seed = gr.Number(value=-1, label="Seed (-1 for random)")
                        guidance_scale = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="Guidance Scale")
                        flow_shift = gr.Slider(0.0, 10.0, value=7.0, step=0.1, label="Flow Shift")
                        embedded_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.5, label="Embedded Guidance Scale")
                show_advanced.change(fn=lambda x: gr.Row(visible=x), inputs=[show_advanced], outputs=[advanced_row])
                
                gr.Markdown("""
                ### Memory Usage Tips:
                1. Start with 540p resolution for testing
                2. Use shorter video length (65f) initially
                3. Increase resolution/length if generation is stable
                4. 720p resolutions recommended for 64GB+ RAM only
                """)
                
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                output = gr.Video(label="Generated Video")
                error_output = gr.Markdown(visible=False)
        
        def handle_generation(*inputs):
            try:
                video_path = generate_video(model, *inputs)
                return video_path, gr.Markdown(visible=False)
            except Exception as e:
                error_msg = str(e)
                return None, gr.Markdown(error_msg, visible=True)
        
        generate_btn.click(
            fn=handle_generation,
            inputs=[
                prompt,
                resolution,
                video_length,
                seed,
                num_inference_steps,
                guidance_scale,
                flow_shift,
                embedded_guidance_scale
            ],
            outputs=[output, error_output]
        )
    
    return demo

if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_name = os.getenv("SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", "8081"))
    args = parse_args()
    print(args)
    demo = create_demo(args.model_base, args.save_path)
    demo.launch(server_name=server_name, server_port=server_port)
