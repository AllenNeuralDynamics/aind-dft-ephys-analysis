# hash:sha256:5561d685f82529e37f7a7c159f3e10f3872295206f208ef8cf78ad37f98a76bc
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/c1-vscode:4.20.0-pylint2023.10.1-copilot1.143.601

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get purge -y --autoremove software-properties-common \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
        software-properties-common=0.99.22.9 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    aind-mri-utils==0.3.0 \
    aind-spurious-correlation==0.0.4 \
    antspyx==0.5.3 \
    hdmf-zarr==0.11.0 \
    ipykernel==6.29.5 \
    jupyter==1.1.1 \
    matplotlib==3.10.0 \
    numpy==2.2.2 \
    pandas==2.2.3 \
    pynwb==2.8.3 \
    scikit-learn==1.6.1 \
    scipy==1.15.1 \
    spikeinterface[full]==0.102.2 \
    statsmodels==0.14.4 \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis.git@5543aa88f3585200e4dc896c1a366de3803c0166#egg=aind-dynamic-foraging-basic-analysis \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-data-utils.git@c2984af731e2d85c250091cfa3cee95c247e5758#egg=aind-dynamic-foraging-data-utils
