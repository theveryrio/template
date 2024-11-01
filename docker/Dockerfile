# Author: Chanho Kim <theveryrio@gmail.com>
# Use the specified Ubuntu base image
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.9-cuda11.1.1

# Set the timezone to Asia/Seoul (Korean Standard Time)
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && echo "Asia/Seoul" > /etc/timezone

# Set environment variables for user
ARG USERNAME
ARG USER_UID
ARG USER_GID

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

# Expose necessary ports
EXPOSE 8888
EXPOSE 8080
EXPOSE 6006
EXPOSE 5000

# Update and install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
apt-utils \
    build-essential \
    curl \
    wget \
    git \
    vim \
    sudo \
    ffmpeg \
    libsm6 \
    libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create the user with provided UID and GID and add to sudo group
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME

# Change to the created user
USER $USERNAME
WORKDIR /home/$USERNAME

# Install Python libraries
RUN pip install --upgrade pip
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install \
    pytorch-lightning==2.0.9 \
    lightning==2.0.9 \
    segmentation-models-pytorch==0.3.3 \
    onnx==1.16 \
    onnxruntime==1.14.1 \
    hydra-core==1.3.2 \
    hydra-optuna-sweeper==1.2.0 \
    mlflow==2.16.2 \
    seaborn==0.13.2 \
    openpyxl==3.1.5 \
    albumentations==1.3.1 \
    opencv-python==4.9.0.80 \
    jupyter==1.1.0

# Set the global email address and username for Git commits
RUN git config --global user.email "theveryrio@gmail.com"
RUN git config --global user.name "Chanho Kim"

# Clone the repository
RUN git clone https://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@github.com/theveryrio/template.git

# Specify the command to run the application
CMD ["./template/docker/run_services.sh"]
