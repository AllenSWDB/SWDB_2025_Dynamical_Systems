# hash:sha256:5c6a11deb1d40e55f6704c812cdcff435241594cfc259a0d1a8f362c675c036c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    aind-analysis-arch-result-access==0.7.2 \
    aind-behavior-gym==0.4.3 \
    aind-dynamic-foraging-models==0.12.0 \
    hdmf-zarr==0.11.3 \
    imageio==2.37.0 \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    pandas==2.3.1 \
    pydantic==2.11.7 \
    pynwb==3.1.1 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    statsmodels==0.14.5

COPY postInstall /
RUN /postInstall
