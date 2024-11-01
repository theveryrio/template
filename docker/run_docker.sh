#!/bin/bash
# Author: Chanho Kim <theveryrio@gmail.com>

# Define the image name
IMAGE_NAME="dev_image"
CONTAINER_NAME="dev_container"

# Function to run the Docker container with dynamic port binding
run_docker()
{
    # Receive ports as an array
    local ports=("$@")
    local port_options=""

    # Convert each port into -p port:port format
    for port in "${ports[@]}"; do
        port_options+="-p $port:$port "
    done

    docker run -it --rm \
    --gpus all \
    --ipc=host \
    --name $CONTAINER_NAME \
    $port_options \
    $VOLUME_MOUNTS \
    $IMAGE_NAME
}

# Parse command-line arguments
VOLUME_MOUNTS=""
while getopts "v:" opt; do
    case ${opt} in
        v )
            VOLUME_MOUNTS+="-v ${OPTARG} "
            ;;
        \? )
            echo "Usage: $0 [-v <volume_mount>]..."
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

run_docker 8888 8080 6006 5000
