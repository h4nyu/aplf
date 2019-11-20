FROM nvidia/cuda:10.1-cudnn7-runtime


MAINTAINER Xinyaun Yao <yao.xinyuan@gmail.com>

ENV LANC=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /srv
COPY ./ /srv
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common curl inotify-tools \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get install python3.7 python3.7-distutils -y \
    && ln -s /usr/bin/python3.7 /usr/bin/python \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && rm get-pip.py \
    && pip install -e .[dev] \
    && apt-get purge -y software-properties-common curl \
    && apt-get autoremove -y


