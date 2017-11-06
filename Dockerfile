FROM nvdl.githost.io:4678/dgx/cuda:9.0-cudnn7-devel-ubuntu16.04--18.01

ENV KALDI_VERSION 5.2
LABEL com.nvidia.kaldi.version="${KALDI_VERSION}"
ENV NVIDIA_KALDI_VERSION 18.01

RUN apt-get update && apt-get install -y --no-install-recommends \
        bzip2 \
        gawk \
        gzip \
        libatlas3-base \
        python-dev \
        subversion \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kaldi

COPY tools tools
RUN cd tools/ && \
    make -j"$(nproc)" && \
    make clean

# Copy remainder of source code
COPY . .

RUN cd src/ && \
    ./configure --shared && \
    make -j"$(nproc)" depend && \
    make -j"$(nproc)" && \
    make -j"$(nproc)" ext && \
    ldconfig

ENV PYTHONPATH $PYTHONPATH:/usr/local/python

WORKDIR /workspace
RUN chmod -R a+w /workspace

COPY nvidia_entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/nvidia_entrypoint.sh"]

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
ARG NVIDIA_BUILD_REF
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"
