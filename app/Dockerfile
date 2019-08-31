FROM nvidia/cuda:10.0-base

MAINTAINER Xinyaun Yao <yao.xinyuan@gmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    g++


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=3.7

ENV PATH /opt/conda/bin:$PATH
RUN pip install pipenv
WORKDIR /srv
COPY ./ /srv
RUN pipenv install

