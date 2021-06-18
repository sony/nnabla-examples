FROM nnabla/nnabla-ext-cuda-multi-gpu:py38-cuda100-multi-gpu-ubuntu18-v1.8.0

ARG PIP_INS_OPTS
ARG PYTHONWARNINGS
ARG CURL_OPTS
ARG WGET_OPTS

LABEL maintainer="Kauki.Yoshiyama@sony.com"

RUN apt-get update
RUN apt-get install -y \
  less \
  libgl1-mesa-glx \
  libgomp1

ENV HTTP_PROXY ${http_proxy}

RUN pip install ${PIP_INS_OPTS} --proxy ${HTTP_PROXY} open3d scikit-image scipy tqdm


