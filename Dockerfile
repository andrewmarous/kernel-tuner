FROM nvidia/cuda:12.8-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    clang \
    clang-tools \
    bear \
    cmake \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV CUDA_PATH=/usr/local/cuda
ENV PATH="${CUDA_PATH}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]
