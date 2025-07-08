# hash:sha256:0b6cde85311a2ae4e70ffa93fdc14bdb7b3b81ea586e51a431345f1d4c1f5d85
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    aind-behavior-gym \
    matplotlib \
    numpy
