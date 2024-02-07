ARG PYTORCH="2.1.0"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

RUN apt-get update && \
    apt-get install -y tzdata && \
    apt install -y \
      tzdata \
      git \
      python3-gdal \
      python3-pip \
      make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Need to peg diffusers to @0.26.0.dev0
RUN pip install -U \
      rasterio \
      git+https://github.com/huggingface/diffusers \
      xformers==0.0.18 \
      accelerate>=0.22.0 \
      transformers>=4.25.1 \
      torchvision \
      peft \
      datasets \
      matplotlib \ 
      ftfy \
      tensorboard \
      bitsandbytes \
      scipy \
      Jinja2
      

COPY ./src /main/src
COPY ./scripts /main/scripts
COPY ./Makefile /main/Makefile

# Copy accelerate config file into the default location
COPY ./config/accelerate_config.yaml /root/.cache/huggingface/accelerate/default_config.yaml 

# Hack to get xformer memory efficient attention working
COPY ./src/attention_processor.py /opt/conda/lib/python3.10/site-packages/diffusers/models/attention_processor.py

# Set HG cache location so we don't have to redownload models every bash session
ARG CACHE_DIR=/main/data/.cache/huggingface
ENV HF_CACHE_DIR=${CACHE_DIR}
ENV HF_HOME=${CACHE_DIR}
ENV HF_DATASETS_CACHE=${CACHE_DIR}/data
ENV TORCH_HOME=${CACHE_DIR}

WORKDIR /main