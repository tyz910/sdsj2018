FROM ubuntu:18.04

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
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        python3.6 \
        python3.6-dev \
        python3.6-venv && \
    pip3 install --no-cache-dir --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Vowpal Wabbit
ENV PATH="/opt/vowpal_wabbit/utl:${PATH}" \
    CPLUS_INCLUDE_PATH=/usr/lib/jvm/java-8-openjdk-amd64/include/linux:/usr/lib/jvm/java-1.8.0-openjdk-amd64/include
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libboost-python-dev\
        libboost-program-options-dev \
        zlib1g-dev \
        openjdk-8-jdk && \
    rm -rf /var/lib/apt/lists/* &&\
    cd opt && git clone git://github.com/JohnLangford/vowpal_wabbit.git && cd vowpal_wabbit && make && make install && \
    cd python && python3 setup.py install

# H2O AutoML
RUN mkdir /tmp/h2o && cd /tmp/h2o && \
    wget http://h2o-release.s3.amazonaws.com/h2o/rel-wright/9/h2o-3.20.0.9.zip && \
    unzip -j h2o-3.20.0.9.zip && \
    pip3 install h2o-3.20.0.9-py2.py3-none-any.whl && \
    rm -rf /tmp/h2o

# Python packages
RUN pip3 install --no-cache-dir --upgrade \
    pandas \
    jupyter \
    lightgbm \
    catboost \
    xgboost \
    hyperopt \
    Boruta \
    category_encoders
