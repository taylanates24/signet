FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV DEBIAN_FRONTEND=noninteractive

ENV MPLBACKEND=agg

RUN apt-get update && \
        apt-get install -y \
        git \
        wget \
        unzip \
        vim \
        zip \
        curl \
        yasm \
        pkg-config \
        nano \
        tzdata \
        ffmpeg \
        libgtk2.0-dev \
        libgl1-mesa-glx \
        software-properties-common && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get upgrade -y libstdc++6 && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip --no-cache-dir install \
      Cython==0.29.21

RUN pip --no-cache-dir install \
      numpy==1.23.1 \
	matplotlib==3.7.1 \
	tqdm==4.65.0 \
	imageio==2.35.1 \
      pillow==9.2.0 \
	opencv-python==4.5.5.64 \
      opencv-python-headless==4.5.5.64 \
	tensorboard==2.9.0 \
	pyyaml==6.0 \
      pytorch-lightning==1.9.2 \
      typeguard==4.3.0 \
      visualdl==2.2.0 \
      shapely==2.0.4 \
      scipy==1.10.1 \
      terminaltables==3.1.10 \
      setuptools==65.5.1 \
      scikit-image==0.21.0 \
      tensorboardX==2.6.2.2 \
      paddle2onnx==1.3.1 \
      timm==1.0.14 \
      imgaug==0.4.0 \
      onnxruntime==1.19.2 \
      clearml==1.17.1 \
      optuna==4.2.1 \
      einops==0.8.1 \
      pyiqa==0.1.13 \
      faster-coco-eval==1.6.5 \
      calflops==0.3.2 \
      transformers==4.37.2 \
      hydra-core==1.3.2 \
      omegaconf==2.3.0 \
      pycuda==2025.1

RUN pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/cudnnin/stable.html 
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
       cd torch2trt && python3 setup.py install && \
       cd .. && \
       rm -rf torch2trt


RUN pip --no-cache-dir install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN ln -sf /usr/share/zoneinfo/Turkey /etc/localtime

