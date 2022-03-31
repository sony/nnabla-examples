FROM nnabla/nnabla-ext-cuda-multi-gpu:py38-cuda110-mpi3.1.6-v1.20.0

ENV HTTP_PROXY ${http_proxy}
ENV HTTPS_PROXY ${https_proxy}

USER root

RUN apt-get update
RUN apt-get install -y libsndfile1 git sox
RUN apt-get install -y python-dev python3.8-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip

RUN pip install --upgrade pip
RUN pip install tqdm librosa numba==0.48.0 matplotlib sox g2p_en pyworld tgt
RUN pip install tensorboard tensorboardX
RUN pip install torch torchvision

# for development
RUN pip install flake8 pycodestyle pytest pytest-cov
