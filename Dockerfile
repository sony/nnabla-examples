# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
ARG CUDA_VER=11.0
ARG CUDNN_VER=8

FROM nvidia/cuda:${CUDA_VER}-cudnn${CUDNN_VER}-runtime-ubuntu18.04

ARG PIP_INS_OPTS
ARG PYTHONWARNINGS
ARG CURL_OPTS
ARG WGET_OPTS
ARG APT_OPTS

ARG PYTHON_VER=3.8
ARG CUDA_VER=11.0
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
    && CUDA_VER_NDOT=`echo $CUDA_VER | sed 's/\.//g'` \
    && pip install ${PIP_INS_OPTS} nnabla-ext-cuda${CUDA_VER_NDOT}-nccl2-mpi2-1-1

RUN umask 0 \
    && CUDA_VER_NDOT=`echo $CUDA_VER | sed 's/\.//g'` \
    && pip install ${PIP_INS_OPTS} --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda${CUDA_VER_NDOT}

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VER} 0
