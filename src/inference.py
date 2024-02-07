from diffusers import  DiffusionPipeline
from peft import LoraConfig
import torch
import utils
import numpy as np
import argparse
from pathlib import Path

import os
os.environ['TRANSFORMERS_CACHE'] = '/wokspace/data/transformers_cache'


def parse_args():
    
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument("--inference_steps", type=int, default=30, help="Number of steps taken during inference.")

    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )

    parser.add_argument("--output_image_n", type=int, default=10, help="Number of images to output.")

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    cwd = utils.cwd()
    base = cwd / "data"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet_lora_config = LoraConfig(
        r=4, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )

    # Final inference
    # Load previous pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(device)

    # load attention processors
    pipeline.load_lora_weights(args.output_dir)

    generator = torch.Generator(device=device)
    generator = generator.manual_seed(args.seed)

    out_dir = Path(args.output_dir).parent.parent / "output" / "lora_rs"
    out_dir.mkdir(exist_ok=True, parents=True)
    for i in range(args.output_image_n):
        img = pipeline(args.validation_prompt, num_inference_steps=args.inference_steps, generator=generator).images[0]
        img = np.array(img)

        utils.save_img(img, out_dir / f"inference_{i}.jpg")

