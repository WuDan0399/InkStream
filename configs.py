import torch
import os


class defaultConfigs:
    batch_sizes = [100] #[1, 10, 100, 1000, 10000]
    num_samples = [1] #[100, 100, 10, 10, 1]

loader_configs = {
    "batch_size": 16,
    "num_workers": 4,
}
root = os.getenv("DYNAMIC_GNN_ROOT")
# root = "/home/dan/GNN/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
