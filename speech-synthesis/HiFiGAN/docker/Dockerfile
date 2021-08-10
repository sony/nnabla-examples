FROM nnabla/nnabla-ext-cuda-multi-gpu:py38-cuda110-mpi3.1.6-v1.20.1

ENV HTTP_PROXY ${http_proxy}
ENV HTTPS_PROXY ${https_proxy}

USER root

RUN apt-get update
RUN apt-get install -y libsndfile1

RUN pip install --upgrade pip
RUN pip install tqdm librosa numba==0.48.0 matplotlib
RUN pip install tensorboard tensorboardX