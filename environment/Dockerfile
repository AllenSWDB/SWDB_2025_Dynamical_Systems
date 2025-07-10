# hash:sha256:793fb077b591ab308e9262c88a70eaa4b31a08c9da401bc58235c0177364cb54
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
    scipy==1.13.1

COPY postInstall /
RUN /postInstall
