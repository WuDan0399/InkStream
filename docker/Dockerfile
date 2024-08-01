FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    git \
    patch \
    && rm -rf /var/lib/apt/lists/*

# Set PATH to include conda
ENV PATH /opt/conda/bin:$PATH

# Create the 'baseline' environment
RUN conda create -n baseline python=3.8
ENV PATH /opt/conda/envs/baseline/bin:$PATH
RUN conda install -n baseline pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
RUN conda install -n baseline numpy -c conda-forge
RUN pip install torch-geometric pynvml

# Create the 'inkstream' environment
RUN conda create -n inkstream python=3.8
ENV PATH /opt/conda/envs/inkstream/bin:$PATH
RUN conda install -n inkstream pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
RUN conda install -n inkstream numpy -c conda-forge
RUN pip install torch-geometric pynvml


#Clone the InkStream repository
RUN git clone https://github.com/WuDan0399/gnn_for_streaming_graphs.git /app/gnn_for_streaming_graphs
# Set the working directory
WORKDIR /app/gnn_for_streaming_graphs

# Apply the patch to torch_geometric in the 'inkstream' environment
RUN /bin/bash -c "source activate inkstream && \
    cd /opt/conda/envs/inkstream/lib/python3.8/site-packages/torch_geometric/nn/conv/ && \
    patch gcn_conv.py < /app/gnn_for_streaming_graphs/docker/gcn_conv.patch && \
    patch sage_conv.py < /app/gnn_for_streaming_graphs/docker/sage_conv.patch && \
    patch gin_conv.py < /app/gnn_for_streaming_graphs/docker/gin_conv.patch"

# Set the default environment to inkstream
RUN echo "source activate inkstream" > ~/.bashrc

