import argparse
import torch
import os
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)

def preprocess_text_encoder_tokenizer(args):
    try:
        # Disable CUDA and MPS
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.enabled = False
        
        processor = AutoProcessor.from_pretrained(args.input_dir)
        print("Using CPU for preprocessing...")
        
        # Load model with CPU-only settings
        model = LlavaForConditionalGeneration.from_pretrained(
            args.input_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=None  # Don't use any device mapping
        )
        model = model.cpu()  # Ensure model is on CPU

        print(f"Saving language model to {args.output_dir}...")
        model.language_model.save_pretrained(
            f"{args.output_dir}",
            safe_serialization=True  # Use safe serialization to avoid GPU tensors
        )
        
        print(f"Saving tokenizer to {args.output_dir}...")
        processor.tokenizer.save_pretrained(
            f"{args.output_dir}"
        )
        
        print(f"Successfully saved model and tokenizer to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The path to the llava-llama-3-8b-v1_1-transformers.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="The output path of the llava-llama-3-8b-text-encoder-tokenizer."
        "if '', the parent dir of output will be the same as input dir.",
    )
    args = parser.parse_args()

    if len(args.output_dir) == 0:
        args.output_dir = "/".join(args.input_dir.split("/")[:-1])

    preprocess_text_encoder_tokenizer(args)
