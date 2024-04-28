# Use the L4T base image from NVIDIA
FROM nvcr.io/nvidia/l4t-base:r32.7.1

ENV DEBIAN_FRONTEND=noninteractive
ENV OPENBLAS_CORETYPE=ARMV8

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-base \
    libcap-dev \
    libopenmpi-dev \
    gfortran \
    git \
    wget \
    python3-dev \
    python3-pip \
    python3-opencv \
    libopenblas-base \
    libopenmpi-dev \
    libomp-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libopenblas-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    curl

# Installing CMAKE
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.0/cmake-3.22.0-Linux-aarch64.tar.gz --no-check-certificate && \
    tar -zxvf cmake-3.22.0-Linux-aarch64.tar.gz && \
    mv cmake-3.22.0-linux-aarch64 /opt/cmake-3.22.0 && \
    ln -s /opt/cmake-3.22.0/bin/cmake /usr/local/bin/cmake && \
    cmake --version && \
    rm cmake-3.22.0-Linux-aarch64.tar.gz

WORKDIR /root
COPY ./requirements.edge.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel scikit-build Cython pybind11
RUN python3 -m pip install -r requirements.edge.txt --only-binary :all: --no-binary pycocotools

# Installing torch & torchvision (Prebuilt wheels from TX2)
COPY ./nvidia_tx2 ./nvidia_tx2
RUN apt-get install -y 
RUN python3 -m pip install ./nvidia_tx2/torch-1.8.0-cp36-cp36m-linux_aarch64.whl    
RUN python3 -m pip install ./nvidia_tx2/torchvision-0.9.0-cp36-cp36m-linux_aarch64.whl

WORKDIR /root/myapp
