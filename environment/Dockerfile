# hash:sha256:e9027c640ce457c35da859a252b2156aae42872172bcbd4d6d29882d8dd5a6da
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    aind-behavior-gym==0.4.3 \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    pandas==2.3.1 \
    pydantic==2.11.7 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    statsmodels==0.14.5

COPY postInstall /
RUN /postInstall
