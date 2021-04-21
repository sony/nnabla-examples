ARG CUDA_VER=10.0
ARG CUDNN_VER=7

FROM nvidia/cuda:${CUDA_VER}-cudnn${CUDNN_VER}-runtime-ubuntu18.04

ARG PIP_INS_OPTS
ARG PYTHONWARNINGS
ARG CURL_OPTS
ARG WGET_OPTS
ARG APT_OPTS

ARG PYTHON_VER=3.7
ARG CUDA_VER=10.0
ARG DALI_VER=0.18
ENV PATH /opt/miniconda3/bin:$PATH
ENV OMP_NUM_THREADS 1

RUN eval ${APT_OPTS} && apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    curl \
    libopenmpi-dev \
    openmpi-bin \
    ssh \
    libglib2.0-0 \
    libgl1-mesa-glx \
    python${PYTHON_VER} \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VER} 0

RUN pip3 install ${PIP_INS_OPTS} --upgrade pip
RUN pip install ${PIP_INS_OPTS} wheel setuptools
RUN pip install ${PIP_INS_OPTS} opencv-python || true
RUN pip install ${PIP_INS_OPTS} jupyter

RUN umask 0 \
    && pip install ${PIP_INS_OPTS} nnabla-ext-cuda`echo $CUDA_VER | sed 's/\.//g'`-nccl2-mpi2-1-1

RUN umask 0 \
    && pip install ${PIP_INS_OPTS} --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/${CUDA_VER} nvidia-dali==${DALI_VER}

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VER} 0
