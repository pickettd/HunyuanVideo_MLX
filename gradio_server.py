import os
import gradio as gr
import numpy as np
from PIL import Image
import torch
from loguru import logger

from hyvideo.inference import HunyuanVideo
from hyvideo.utils.file_utils import save_video_grid

def create_interface():
    # Initialize model with MLX pipeline
    model = HunyuanVideo.from_pretrained("ckpts")
    
    def generate_video(
        prompt,
        video_length=16,
        video_width=960,
        video_height=540,
        seed=42,
        guidance_scale=7.0,
        num_inference_steps=25,
        progress=gr.Progress()
    ):
        """Generate video using MLX pipeline"""
        try:
            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Generate video
            outputs = model.predict(
                prompt=prompt,
                height=video_height,
                width=video_width,
                video_length=video_length,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed
            )
            
            # Save video
            save_path = os.path.join("results", f"{seed}.mp4")
            os.makedirs("results", exist_ok=True)
            
            # Convert to PIL images for display
            video = outputs.videos[0]
            video = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            frames = [Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()) for frame in video]
            
            # Save video grid
            save_video_grid(frames, save_path, fps=8)
            
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating video: {str(e)}")
            raise gr.Error(str(e))
    
    # Create Gradio interface
    with gr.Blocks() as interface:
        gr.Markdown("# HunyuanVideo MLX Demo")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                
                with gr.Row():
                    video_length = gr.Slider(
                        label="Video Length",
                        minimum=16,
                        maximum=128,
                        step=1,
                        value=16
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=20.0,
                        step=0.5,
                        value=7.0
                    )
                
                with gr.Row():
                    video_width = gr.Slider(
                        label="Width",
                        minimum=256,
                        maximum=1024,
                        step=64,
                        value=960
                    )
                    video_height = gr.Slider(
                        label="Height",
                        minimum=256,
                        maximum=1024,
                        step=64,
                        value=540
                    )
                
                with gr.Row():
                    seed = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0
                    )
                    steps = gr.Slider(
                        label="Inference Steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=25
                    )
                
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                result = gr.Video(label="Generated Video")
        
        # Connect components
        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                video_length,
                video_width,
                video_height,
                seed,
                guidance_scale,
                steps
            ],
            outputs=result
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0")
