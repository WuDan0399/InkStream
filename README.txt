# Preparation:
1. Create docker container with conda environments: baseline and inkstream.
    cd docker
    docker build -t inkstream_image .
    docker run --name inkstream_container --gpus all -it inkstream_image
2. Run InkStream:
    `python inkstream_gcn.py --dataset cora --model GCN --save_int --aggr min --perbatch 100 --stream mix`