FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    iproute2 \
    git \
    python3 \
    python3-pip

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda init bash
RUN conda update -n base -c defaults conda

RUN conda create --name yolov6_env python=3.8
RUN conda activate yolov6_env
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

RUN git clone https://github.com/hectorandac/Gated-YOLOv6

CMD ["/bin/bash"]
