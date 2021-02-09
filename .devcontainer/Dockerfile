FROM python:3.8-buster
LABEL maintainer="benjamin.wilhelm@uni-konstanz.de"

ARG USERNAME=vsdocker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Install essentials
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get -y install --no-install-recommends git vim zsh tmux less

# Runtime dependencies
COPY ./requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

# Develop dependencies
RUN pip install flake8 mypy pytest autopep8 jupyter matplotlib
