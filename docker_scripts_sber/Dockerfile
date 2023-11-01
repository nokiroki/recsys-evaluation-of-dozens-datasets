FROM cr.msk.sbercloud.ru/aicloud-base-images-test/cuda11.7-torch2:8aa47206-663069


USER root
# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

ARG DEBIAN_FRONTEND=noninteractive
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    sox \
    libsox-dev \
    libsox-fmt-all \
    build-essential \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    unzip \
 && rm -rf /var/lib/apt/lists/*

# Java install
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

# Create a working directory
RUN mkdir /app
WORKDIR /app

ARG DOCKER_NAME
ARG DOCKER_USER_ID
ARG DOCKER_GROUP_ID

RUN groupadd -g $DOCKER_GROUP_ID $DOCKER_NAME

RUN adduser --disabled-password --uid $DOCKER_USER_ID --gid $DOCKER_GROUP_ID --gecos '' --shell /bin/bash $DOCKER_NAME \
 && chown -R $DOCKER_NAME:$DOCKER_NAME /app
RUN echo "$DOCKER_NAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-$DOCKER_NAME
USER $DOCKER_NAME

ENV HOME=/home/$DOCKER_NAME

RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Set up the Conda environment
# ENV CONDA_AUTO_UPDATE_CONDA=false \
#     PATH=$HOME/miniconda/bin:$PATH
# COPY environment.yml /app/environment.yml
# RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
#  && chmod +x ~/miniconda.sh \
#  && ~/miniconda.sh -b -p ~/miniconda \
#  && rm ~/miniconda.sh \
#  && conda env create --file=environment.yml \
#  && rm /app/environment.yml \
#  && conda clean -ya

# SHELL ["conda", "run", "-n", "base2", "/bin/bash", "-c"]

# RUN python -c "import hydra"

# ENTRYPOINT ["conda", "activate", "base2"]

RUN pip install hydra-core \
    && pip install implicit \
    && pip install ezSASRec \
    && pip install surprise \
    && pip install replay-rec \
    && pip install recbole \
    && pip install baycomp \