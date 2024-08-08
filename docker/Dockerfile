# Use the NVIDIA CUDA base image with Ubuntu and Conda installed
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set the environment variable to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -a -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Update PATH environment variable
ENV PATH /opt/conda/bin:$PATH


# Copy the environment.yml file
COPY environment.yml /app/environment.yml

# Create the 'baseline' conda environment
RUN conda env create -f /app/environment.yml -n baseline

# Create the 'inkstream' conda environment by copying 'baseline'
RUN conda create --name inkstream --clone baseline
#RUN pip install torch_geometric pynvml
#RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# Install additional packages in both environments
RUN conda run -n baseline pip install torch_geometric pynvml && \
    conda run -n baseline pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

RUN conda run -n inkstream pip install torch_geometric pynvml && \
    conda run -n inkstream pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

#Clone the InkStream repository
RUN git clone https://github.com/WuDan0399/gnn_for_streaming_graphs.git /app/gnn_for_streaming_graphs
# Set the working directory
WORKDIR /app/gnn_for_streaming_graphs

# Apply the patch to torch_geometric in the 'inkstream' environment
RUN /bin/bash -c "cd /opt/conda/envs/inkstream/lib/python3.8/site-packages/torch_geometric/nn/conv/ && \
    patch gcn_conv.py < /app/gnn_for_streaming_graphs/docker/gcn_conv.patch && \
    patch sage_conv.py < /app/gnn_for_streaming_graphs/docker/sage_conv.patch && \
    patch gin_conv.py < /app/gnn_for_streaming_graphs/docker/gin_conv.patch && \
    "

# Activate the 'baseline' environment by default
RUN echo "conda activate inkstream" >> ~/.bashrc
RUN echo "export DYNAMIC_GNN_ROOT=/app/gnn_for_streaming_graphs" >> ~/.bashrc

# Set the default shell to use conda's bash
SHELL ["conda", "run", "-n", "inkstream", "/bin/bash", "-c"]

# Set the default command to run when starting the container
CMD ["bash"]
