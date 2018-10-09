FROM ubuntu:16.04

ENV LANG=C.UTF-8

# Common packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        vim \
        wget \
        curl \
        git \
        zip \
        unzip && \
    rm -rf /var/lib/apt/lists/*

# Python 3.6
RUN add-apt-repository -y ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        python3.6 \
        python3.6-dev \
        python3.6-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2 && \
    pip3 install --no-cache-dir --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip3 install --no-cache-dir --upgrade \
    pandas \
    jupyter \
    lightgbm
