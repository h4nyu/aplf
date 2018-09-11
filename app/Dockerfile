FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER Xinyaun Yao <yao.xinyuan@gmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    bzip2 \
    graphviz \
    g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=3.6  && \
    /opt/conda/bin/conda install -c pytorch magma-cuda90 && \
    /opt/conda/bin/conda clean -ya


ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
RUN pip install --upgrade pip
WORKDIR /srv
COPY ./requirements.txt /srv/requirements.txt
COPY ./dev-requirements.txt /srv/dev-requirements.txt

RUN pip install -r requirements.txt --no-cache-dir  && \
    pip install -r dev-requirements.txt  --no-cache-dir
COPY ./ /srv
RUN python setup.py develop
