#!/bin/bash
# Author: Chanho Kim <theveryrio@gmail.com>

VOLUMES=$1
SCREEN_NAME="dev_screen"
CONTAINER_NAME="dev_container"

# Check if a screen session named '${SCREEN_NAME}' already exists
if screen -list | grep -q "\.${SCREEN_NAME}"; then
    echo "A screen session named '${SCREEN_NAME}' already exists. Terminating it..."
    screen -S $SCREEN_NAME -X quit
    docker stop $CONTAINER_NAME
fi

# Create a new screen session named '${SCREEN_NAME}'
echo "Creating a new screen session named '${SCREEN_NAME}'..."
screen -S $SCREEN_NAME -dm bash

# Send commands to the '${SCREEN_NAME}' screen session
screen -S $SCREEN_NAME -X stuff "cd $HOME/template/docker\n"
screen -S $SCREEN_NAME -X stuff "./build.sh\n"
screen -S $SCREEN_NAME -X stuff "./run_docker.sh $VOLUMES\n"
