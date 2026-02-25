# hash:sha256:ec1b490392664b6c81ea8174f0a6590c12c37d5decefcdd9637b55344c9b0700
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/c1-vscode:4.20.0-pylint2023.10.1-copilot1.143.601

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
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

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip3 install -U --no-cache-dir \
    aind-analysis-arch-result-access==1.0.0 \
    aind-mri-utils==0.3.0 \
    aind-spurious-correlation==0.0.6 \
    antspyx==0.5.3 \
    hdmf-zarr==0.11.0 \
    ipykernel==6.29.5 \
    jupyter==1.1.1 \
    matplotlib==3.10.0 \
    matplotlib-venn==1.1.2 \
    numpy==2.2.2 \
    pandas==2.2.3 \
    pynwb==2.8.3 \
    scikit-learn==1.6.1 \
    scipy==1.15.1 \
    spikeinterface[full]==0.102.2 \
    statsmodels==0.14.4 \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis.git@5543aa88f3585200e4dc896c1a366de3803c0166#egg=aind-dynamic-foraging-basic-analysis \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-data-utils.git@c2984af731e2d85c250091cfa3cee95c247e5758#egg=aind-dynamic-foraging-data-utils \

RUN pip install git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-models.git@update_CTT#egg=aind-dynamic-foraging-models

#ADD "https://github.com/coder/code-server/releases/download/v4.95.3/code-server-4.95.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
#RUN cd /.code-server \
#    && tar -xvf code-server.tar.gz \
#    && rm code-server.tar.gz \
#    && ln -s /.code-server/code-server-4.95.3-linux-amd64/bin/code-server  /usr/bin/code-server

#RUN mkdir -p /.vscode/extensions \
#    && code-server --extensions-dir="/.vscode/extensions" --install-extension REditorSupport.R \
#    && code-server --extensions-dir="/.vscode/extensions" --install-extension continue.continue \
#    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python \
#    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter \
#    && code-server --extensions-dir="/.vscode/extensions" --install-extension reageyao.bioSyntax \
#    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev

COPY postInstall /
RUN /postInstall
