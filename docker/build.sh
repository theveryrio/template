#!/bin/bash
# Author: Chanho Kim <theveryrio@gmail.com>

# Get the current user info
USERNAME=$(whoami)
USER_UID=$(id -u)
USER_GID=$(id -g)

# Define the image name
IMAGE_NAME="dev_image"

# Build the Docker image
docker build --build-arg USERNAME=$USERNAME --build-arg USER_UID=$USER_UID --build-arg USER_GID=$USER_GID --label maintainer=$USERNAME -t $IMAGE_NAME .
docker image prune -f
