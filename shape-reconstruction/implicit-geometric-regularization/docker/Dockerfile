# Copyright 2020,2021 Sony Corporation.
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


