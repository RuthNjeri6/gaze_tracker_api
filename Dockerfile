FROM python:3.10.3-slim-bullseye
WORKDIR /app
RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    build-essential \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*
RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r /app/requirements.txt  --upgrade
COPY ./api /app/
EXPOSE 8000
ENTRYPOINT ["uvicorn"]