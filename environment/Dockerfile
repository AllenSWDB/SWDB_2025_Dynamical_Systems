# hash:sha256:35220e0929bf1c60a8764bb77dd213036278f82b2145654a98894a69b5e67657
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    aind-analysis-arch-result-access==0.7.2 \
    aind-behavior-gym==0.4.3 \
    aind-dynamic-foraging-models==0.12.0 \
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
