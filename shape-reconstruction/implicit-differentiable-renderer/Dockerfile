# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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

# This is copy & paste of https://github.com/sony/nnabla-ext-cuda/blob/v1.19.0/docker/py38/cuda110/mpi3.1.6/Dockerfile, but with several differences
# - development version of cuda/cudnn
# - pip/apt install dependencys
# - build softplus cuda

FROM ubuntu:18.04 as openmpi

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        gfortran \
        libibverbs-dev

RUN apt-get install -y --no-install-recommends ca-certificates

RUN mkdir /tmp/openmpi
RUN curl https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.6.tar.bz2 -o /tmp/openmpi/openmpi-3.1.6.tar.bz2
RUN tar Cxvf /tmp/openmpi /tmp/openmpi/openmpi-3.1.6.tar.bz2
RUN cd tmp/openmpi/openmpi-3.1.6 \
    && ./configure \
        --prefix=/opt/openmpi --enable-orterun-prefix-by-default --with-sge \
        CC=gcc \
        CXX=g++ \
        F77=gfortran \
        FC=gfortran \
    && make -j8 \
    && make install

FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bzip2 \
       ca-certificates \
       curl \
       libglib2.0-0 \
       libgl1-mesa-glx \
       python3.8 \
       python3-pip \
       openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 0
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 0

RUN pip3 install --upgrade pip
RUN pip install wheel setuptools
RUN pip install opencv-python || true

RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110 \
    || echo "Skip DALI installation (CUDA=11.0)"

COPY --from=openmpi /opt/openmpi /opt/openmpi
ENV PATH /opt/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH /opt/openmpi/lib:$LD_LIBRARY_PATH


ARG NNABLA_VER
RUN pip install nnabla-ext-cuda110-nccl2-mpi3.1.6==${NNABLA_VER}

# Solve nccl error that No space left on device
# while creating shared memory segment.
RUN echo NCCL_SHM_DISABLE=1 >> /etc/nccl.conf
RUN echo NCCL_P2P_LEVEL=SYS >> /etc/nccl.conf

# Prepare default user
RUN useradd -m nnabla
RUN chown -R nnabla:nnabla /home/nnabla

# Python dependency
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Apt dependency
RUN apt-get update && apt-get install -y libusb-1.0-0

ENV PYTHONPATH /opt/lib:$PYTHONPATH

USER nnabla
WORKDIR /home/nnabla
CMD [ "/bin/bash" ]


