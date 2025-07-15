FROM nvcr.io/nvidia/pytorch:24.12-py3

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
    setuptools \
    numpy==1.26.4 \
	matplotlib==3.9.3 \
	tqdm \
    pillow==11.0.0 \
    opencv-python==4.5.5.64 \
    scipy==1.14.1 \
    scikit-image==0.25.2 \
    onnx \
    transformers==4.49.0 \
    onnxsim==0.4.36 \
    onnxruntime-gpu \
    tensorrt==10.7.0 \
    tensorboard==2.16.2 \
    pycuda

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


RUN ln -sf /usr/share/zoneinfo/Turkey /etc/localtime

