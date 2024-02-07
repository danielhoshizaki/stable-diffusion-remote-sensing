project_name=$(shell basename $(PWD))
pwd=$(shell pwd)

build:
	docker build -t $(project_name) --progress=plain .

bash:
	docker run -it --gpus all --rm --shm-size=24gb -v $(pwd):/main $(project_name) bash


##############################################################
Make commands for running a bash session WITHIN the container

MODEL_NAME="runwayml/stable-diffusion-v1-5"
OUTPUT_DIR="/main/models/lora_rs"
DATASET_NAME="/main/data"
ACCELERATE_CONFIG="/main/config/config.yaml"
PROMPT="satellite image of mountains"
SEED=1430
IMG_SZIE=512

inference:
	python3 src/inference.py \
		--mixed_precision="bf16" \
		--pretrained_model_name_or_path=${MODEL_NAME} \
		--output_dir=${OUTPUT_DIR} \
		--seed=${SEED} \
		--inference_steps=50 \
		--validation_prompt=${PROMPT} \
		--output_image_n=6

train:
	accelerate launch src/train_text_to_image_lora.py \
		--mixed_precision="bf16" \
		--pretrained_model_name_or_path=${MODEL_NAME} \
		--dataset_name=${DATASET_NAME} \
		--dataloader_num_workers=8 \
		--resolution=${IMG_SZIE} --center_crop \
		--random_flip \
		--train_batch_size=2 \
		--gradient_accumulation_steps=8 \
		--max_train_steps=200 \
		--learning_rate=1e-03 \
		--max_grad_norm=1 \
		--lr_scheduler="cosine" \
		--lr_warmup_steps=0 \
		--output_dir=${OUTPUT_DIR} \
		--checkpointing_steps=1000 \
		--enable_xformers_memory_efficient_attention \
		--validation_prompt ${PROMPT} \
		--seed=${SEED}

preload-model:
	python3 scripts/preload_model.py --pretrained_model_name_or_path=${MODEL_NAME}