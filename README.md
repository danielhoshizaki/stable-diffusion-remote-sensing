# Stable Diffusion for Remote Sensing

This repository provides companion code for my blog post: [Stable Diffusion for Remote Sensing: Reality Check ](https://danielhoshizaki.com/stable%20diffusion/remote%20sensing/deep%20learning/2024/01/20/stable-diffusion.html).

## Getting Started

Use Docker and Make to build the Docker container. My system includes two RTX3060 and I have configured `accelerate` to use both. If you are only working with one GPU can change the settings in the `./config/accelerate_config.yaml` file or run the script without `accelerate`.

I've taken Huggingface's `train_text_to_image_lora.py` [script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py) and modified it with a few hacks to get it working in the Docker container. Please note that the container relies on a development version of Huggingface `diffusers` and that there is no way to specify the same development version if Huggingface updates the branch.

Prepare a dataset and create a `./data/train` directory with a metadata.jsonl file and associated training images.

## Usage

Build the Docker container and start a bash session in the container once the building is complete.

```
make build
make bash
```

Within the container, you can use the following make commands to train SD1.5 and generate inference images.

```
make train
make inference
```