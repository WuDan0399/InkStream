# InkStream: Instantaneous GNN Inference on Dynamic Graphs via Incremental Update

PyTorch implementation of optimizing the GNN inference for dynamic graphs via incremental computing. InkStream is able to reduce the inference time to milliseconds for dynamic graphs evolving with minor changes each time.


## Quick Start
1. Create docker container with conda environments: `baseline` and `inkstream`.
    ```bash
    cd docker
    docker build -t inkstream_image .
    docker run --name inkstream_container --gpus all --shm-size=32g -it inkstream_image
    ```
2. Run InkStream:
    ```bash
    python inkstream_<gcn/sage/gin>.py --dataset <cora/PubMed/yelp/reddit/products/papers> --model <GCN/SAGE/GIN> --save_int --aggr <min/max/mean/sum> --perbatch <number of changed edges> --stream <mix/add/delete>
   ```
&nbsp;&nbsp;&nbsp;&nbsp;e.g., `python inkstream_gcn.py --dataset cora --model GCN --save_int --aggr min --perbatch 100 --stream mix`

&nbsp;&nbsp;&nbsp;&nbsp;**Note:**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Inkstream can only be executed in `inkstream` conda environment inside the docker, as we made modifications to the torch_geometric library. Baseline methods should be executed in `baseline` conda environment.

3. Run baseline (k-hop):
    ```bash
    conda activate baseline
    python timing_original.py --dataset cora --model GCN --aggr min --perbatch 100 --stream mix --range affected
    ```
