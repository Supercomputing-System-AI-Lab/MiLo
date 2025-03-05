#!/bin/bash

conda install -y -c conda-forge \
    _libgcc_mutex=0.1 \
    _openmp_mutex=4.5 \
    bzip2=1.0.8 \
    ca-certificates=2025.1.31 \
    ld_impl_linux-64=2.43 \
    libffi=3.4.6 \
    libgcc=14.2.0 \
    libgcc-ng=14.2.0 \
    libgomp=14.2.0 \
    liblzma=5.6.4 \
    libnsl=2.0.1 \
    libuuid=2.38.1 \
    libxcrypt=4.4.36 \
    libzlib=1.3.1 \
    ncurses=6.5 \
    openssl=3.4.1 \
    readline=8.2 \
    tk=8.6.13 \
    libsqlite=3.49.1

# Install CUDA and NVIDIA GPU libraries
conda install -y -c nvidia \
    cuda=12.4.0 \
    cuda-cccl=12.4.99 \
    cuda-command-line-tools=12.4.0 \
    cuda-compiler=12.4.0 \
    cuda-cudart=12.4.99 \
    cuda-cudart-dev=12.4.99 \
    cuda-cupti=12.4.99 \
    cuda-gdb=12.4.99 \
    cuda-libraries=12.4.0 \
    cuda-nvcc=12.4.99 \
    cuda-toolkit=12.4.0 \
    gds-tools=1.9.0.20 \
    libcublas=12.4.2.65 \
    libcufft=11.2.0.44 \
    libcufile=1.9.0.20 \
    libcurand=10.3.5.119 \
    libcusolver=11.6.0.99 \
    libcusparse=12.3.0.142 \
    libnpp=12.2.5.2 \
    nsight-compute=2024.1.0.13

pip install -r requirements.txt



