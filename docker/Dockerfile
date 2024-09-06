# FROM mambaorg/micromamba:bionic-cuda-11.6.2
FROM mambaorg/micromamba:jammy-cuda-12.1.0


COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml

USER root
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git -y

USER $MAMBA_USER
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

WORKDIR "/thesis"

ENTRYPOINT ["/bin/bash"]
