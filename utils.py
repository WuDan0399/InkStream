import subprocess
import argparse
import os
import os.path as osp
import re
import time
from functools import wraps
from itertools import groupby
from operator import attrgetter
from collections import defaultdict
from typing import Tuple, Union, List, Dict, Optional, Callable


import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.loader import NeighborLoader, NeighborSampler
from torch_geometric.typing import OptTensor
from torch_geometric.utils import degree
from torch.utils.data import DataLoader
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data, Batch
from configs import *
import pickle

# for memory usage tracing
import pynvml
from threading import Thread, Event

np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=10)

print_data_loader = True

class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class FakeArgs:
    def __init__(
        self,
        dataset="Cora",
        model="GCN",
        use_gdc=False,
        patience=2,
        aggr="min",
        save_int=True,
        hidden_channels=256,
        epochs=10,
        perbatch=1,
        stream="add",
        interval=5000000,
        it=0,
        range="affected",
        loader="default",
    ):
        self.dataset = dataset
        self.model = model
        self.use_gdc = use_gdc
        self.aggr = aggr
        self.save_int = save_int
        self.hidden_channels = hidden_channels
        self.perbatch = perbatch
        self.stream = stream
        self.interval = interval
        self.it = it
        self.patience = patience
        self.epochs = epochs
        self.range = range
        self.loader = loader


def get_gpu_memory_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2  # Convert to MiB

class GPUMemoryMonitor(Thread):
    def __init__(self, sampling_rate=0.1):
        super().__init__()
        self.sampling_rate = sampling_rate  # Time between samples in seconds
        self.done = Event()
        self.max_memory = 0
        self.memory_usage = []

    def run(self):
        while not self.done.is_set():
            mem_usage = get_gpu_memory_usage()
            self.memory_usage.append(mem_usage)
            self.max_memory = max(self.max_memory, mem_usage)
            time.sleep(self.sampling_rate)

    def stop(self):
        self.done.set()


def replace_arrays_with_shape_and_type(**kwargs):
    """
    For meature_time decorator, in case one of the argument is a tensor\array,
    the measure_time function will print the content of the tensor\array, which is messy.
    :param kwargs:  any function arguments that using measure_time decorator
    :return:  a modified argument dict with tensor\arrays replaced with a descriptive string
    """
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            kwargs[key] = f"numpy.ndarray, shape: {value.shape}, dtype: {value.dtype}"
        elif isinstance(value, torch.Tensor):
            kwargs[key] = f"torch.Tensor, shape: {value.shape}, dtype: {value.dtype}"
    return kwargs


# decorator
def measure_time(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f"Function {func.__name__} {args} {replace_arrays_with_shape_and_type(**kwargs)} Took {total_time:.4f} seconds"
        )

        return result

    return timeit_wrapper


def save(model_state_dict, epoch, acc, name) -> None:
    torch.save(
        model_state_dict,
        osp.join("examples/trained_model", f"{name}_{epoch}_{acc:.3f}.pt"),
    )


def load(model, file_name: str):
    print(f"Loading model {file_name} ...")
    model_path = osp.join(root, "examples", "trained_model", file_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def clean(files: list) -> None:
    if len(files) == 0:
        print("Clean directory, no useless models.")
    else:
        print(f"Useless models found, cleaning {files}.")
        for file in files:
            os.remove(osp.join(root, "examples", "trained_model", file))


def load_tensors_to_dict(root: str, skip: int = 5, postfix: str = "_initial.pt"):
    result = {}
    for path, dirs, files in os.walk(root):
        if files: 
            path_parts = path.split(os.sep)
            sub_dict = result
            for part in path_parts[skip:]:
                if part not in sub_dict:
                    sub_dict[part] = {}
                sub_dict = sub_dict[part]
            for file in files:
                if file.endswith(
                    postfix
                ): 
                    name_parts = file[:-3].split(
                        "_"
                    )
                    sub_dict[name_parts[0]] = torch.load(
                        os.path.join(path, file))
    return result


def add_mask(data: pyg.data.Data) -> None:
    if not hasattr(data, "train_mask"):
        print("Add mask to the dataset.")
        if data.num_nodes < 1000000:  
            train_ratio, test_ratio, validation_ratio = 0.75, 0.1, 0.15 
        else:
            shrink_ratio = 1000000 / data.num_nodes  
            train_ratio, test_ratio, validation_ratio = 0.75 * shrink_ratio, 0.1 * shrink_ratio, 0.15 * shrink_ratio 
        rand_choice = np.random.rand(len(data.y))
        data.train_mask = torch.tensor(
            [True if x < train_ratio else False for x in rand_choice], dtype=torch.bool
        )
        data.test_mask = torch.tensor(
            [
                True if x >= train_ratio and x < train_ratio + test_ratio else False
                for x in rand_choice
            ],
            dtype=torch.bool,
        )
        data.val_mask = torch.tensor(
            [True if x >= train_ratio + test_ratio and x < train_ratio + test_ratio + validation_ratio
             else False for x in rand_choice],
            dtype=torch.bool,
        )


def data_loader(
    data,
    input_nodes=None,
    num_layers: int = 2,
    num_neighbour_per_layer: int = -1,
    separate: bool = True,
    persistent_workers: bool = False,
    loader_type: str = "default"
):
    kwargs = loader_configs
    kwargs["persistent_workers"] = persistent_workers
    global print_data_loader
    if loader_type == "default":
        if separate:
            train_loader = NeighborLoader(
                data,
                num_neighbors=[num_neighbour_per_layer] * num_layers,
                shuffle=True,
                input_nodes=data.train_mask,
                **kwargs,
            )
            val_loader = NeighborLoader(
                data,
                num_neighbors=[num_neighbour_per_layer] * num_layers,
                input_nodes=data.val_mask,
                **kwargs,
            )
            test_loader = NeighborLoader(
                data,
                num_neighbors=[num_neighbour_per_layer] * num_layers,
                input_nodes=data.test_mask,
                **kwargs,
            )
            return train_loader, val_loader, test_loader
        else:
            return NeighborLoader(
                data,
                input_nodes=input_nodes,
                num_neighbors=[num_neighbour_per_layer] * num_layers,
                **kwargs,
            )
    elif loader_type == "quiver":
        if separate:
           raise NotImplementedError
        else:
            seed_loader = torch.utils.data.DataLoader(torch.arange(0, data.num_nodes-1).long(),
                                                       batch_size=loader_configs["batch_size"])
            return seed_loader


def print_dataset(dataset) -> None:
    print(f"Dataset: {dataset}:")
    print("======================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")


def print_data(data: pyg.data.Data) -> None:
    print(data)
    print("==============================================================")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Average node degree: {(2 * data.num_edges) / data.num_nodes:.2f}")
    if hasattr(data, "train_mask"):
        print(f"Number of training nodes: {data.train_mask.sum()}")
        print(
            f"Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}"
        )
    print(f"Contains isolated nodes: {data.has_isolated_nodes()}")
    print(f"Contains self-loops: {data.has_self_loops()}")
    print(f"Is undirected: {data.is_undirected()}")


def general_parser(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("-d", "--dataset", type=str, default="yelp")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--aggr", type=str, default="min")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--use_gdc", action="store_true", help="Use GDC")
    parser.add_argument("--mode",type=str, default="mixed", help="for dgl train mode")
    parser.add_argument("-l", "--nlayers", default=5,
                        type=int, help="number of layers")
    parser.add_argument(
        "-i",
        "--initial",
        default=70.0,
        type=float,
        help="percentage of edges loaded at the begining. [0.0, 100.0]",
    )
    parser.add_argument(
        "-pb",
        "--perbatch",
        default=100,
        type=float,
        help="percentage of edges loaded per batch. [0.0, 100.0]",
    )
    parser.add_argument(
        "--interval",
        default=5000000,
        type=float,
        help="percentage of the latest proportion/number of edges for inference. [0.0, 1.0] or int. Default: 500k edges",
    )
    parser.add_argument(
        "-nb", "--numbatch", default=50, type=int, help="number of batches"
    )
    parser.add_argument(
        "--range",
        default="full",
        type=str,
        help="range of inference [full/affected/mono]",
    )
    parser.add_argument(
        "--use_loader", action="store_true", help="whether to use data loader"
    )
    parser.add_argument(
        "--save_int",
        action="store_true",
        help="whether to save intermediate timing_result during inference",
    )
    parser.add_argument(
        "--stream",
        default="mix",
        type=str,
        help="how edges changes, insertion or deletion or both [add/delete/mix]",
    )
    parser.add_argument(
        "--loader",
        default="default",
        type=str,
        help="loader to use, pyg.Neighborloader by default [default/quiver]",
    )
    parser.add_argument(
        "--id",
        default=-1,
        type=int,
        help="id of the sample to be inferred, only used in multi-threading",
    )
    args = parser.parse_args()
    return args


def timing_sampler(data: pyg.data.Data, args):
    """
    Simulate the streaming graphs. Generate random time information, get the lastest 'interval' edges of data for gnn.
    :param data: PyG.data
    :param args: command arguments
    :param interval: float or int. in (0,1) -> ratio. >1 -> exact size of num_edges
    :return: the lastest 'interval' edges for data.
    """
    if isinstance(data, pyg.data.Data):
        num_edges = data.num_edges
        edge_index = data.edge_index
    else:
        num_edges = data.num_edges()
        edge_index = torch.stack(data.edges())
    print(f"Sampling {args.interval} edges for {args.dataset}")
    folder_path = os.path.join("examples", "creation_time")
    create_directory(folder_path)

    if args.interval > 1:
        interval = args.interval / num_edges
        threshold = 1 - interval
    else:
        interval = args.interval
        threshold = 1 - args.interval

    file_path = os.path.join(
        "examples", "creation_time", f"{args.dataset}_{interval}_{threshold}.pt"
    )

    if os.path.exists(file_path):
        mask = torch.load(file_path)
    else:
        print(f"Generate random time information for dataset {args.dataset}")
        mask = torch.rand(num_edges)
        torch.save(mask, file_path)

    filtered_edge_index = edge_index[:, mask > threshold]
    if isinstance(data, pyg.data.Data):
        data.edge_index = filtered_edge_index
    else: 
        data.remove_edges(torch.arange(data.number_of_edges()))
        data.add_edges(filtered_edge_index[0], filtered_edge_index[1])
    return


def load_available_model(model, args: argparse.Namespace):
    model_name = f"{args.dataset}_{args.model}_{args.aggr}.pt"
    if not os.path.exists(osp.join(root, "examples", "trained_model", model_name)):  # no available model, train from scratch
        print(f"No available model. Please run `python {args.model}.py --dataset {args.dataset} --aggr {args.aggr}`")
        return None
    model = load(model, model_name)
    return model


def count_layers(model):
    count = 0
    for child in model.children():
        if any(
            [
                isinstance(child, cls)
                for cls in vars(pyg.nn.conv).values()
                if isinstance(cls, type)
            ]
        ):
            count += 1
        elif len(list(child.children())) > 0:
            count += count_layers(child)
    return count


def affected_nodes_each_layer(
    edge_dicts: List[Dict[int, torch.Tensor]],
    srcs: Union[list, set],
    depth: int,
    self_loop: bool = False,
) -> defaultdict:
    affected = defaultdict(set)
    affected[0] = set(srcs)
    for i in range(depth):
        affected[i + 1].update(srcs)
        if self_loop:
            affected[i + 1].update(affected[i])
        for node in affected[i]:
            for edge_dict in edge_dicts:
                affected[i + 1].update(edge_dict[node])
    return affected


def get_graph_dynamics(
    tensor: torch.Tensor, batch_size: int, stream: str = "mix"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :param tensor: original edge_index
    :param batch_size: number of streaming edges in a time interval
    :param mode: operations of edges. insertion, deletion or both.
    :return: initial_edges, final_edges, inserted_edges, removed_edges
    """
    assert batch_size < tensor.size(
        1
    ), "batch_size should be smaller than the number of columns in the tensor"

    indices = torch.randperm(tensor.size(1))

    if stream == "add":
        new_tensor = tensor[:, indices[:-batch_size]]
        removed_tensor = tensor[:, indices[-batch_size:]]
        return new_tensor, tensor, removed_tensor, torch.empty((2, 0))

    elif stream == "delete":
        new_tensor = tensor[:, indices[:-batch_size]]
        removed_tensor = tensor[:, indices[-batch_size:]]
        return tensor, new_tensor, torch.empty((2, 0)), removed_tensor

    elif stream == "mix":
        set_A_indices = indices[: batch_size // 2]
        set_B_indices = indices[batch_size // 2: batch_size]
        tensor_A = tensor[:, indices[batch_size // 2:]]
        tensor_B = tensor[
            :, torch.cat((indices[: batch_size // 2], indices[batch_size:]))
        ]
        set_A = tensor[:, set_A_indices]
        set_B = tensor[:, set_B_indices]
        if batch_size == 1 and torch.rand(1) > 0.5:
            set_A, set_B = set_B, set_A
            tensor_A, tensor_B = tensor_B, tensor_A
        return tensor_A, tensor_B, set_A, set_B


def is_large(data):
    return True if data.num_nodes > 50000 else False


def create_directory(path): 
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

def to_dict_wiz_cache(edges:torch.Tensor, dir, filename):
    edge_dict_file_name = os.path.join(dir, filename)
    if os.path.exists(edge_dict_file_name):
        with open(edge_dict_file_name, 'rb') as file:
            edge_dict = pickle.load(file)
    else:
        edge_dict = to_dict(edges)
        with open(edge_dict_file_name, 'wb') as file:
            pickle.dump(edge_dict, file)
    return edge_dict

def to_dict(edges: torch.Tensor):
    edge_dict = defaultdict(list)
    sources, destinations = edges[0].tolist(), edges[1].tolist()
    for i, (source, dest) in enumerate(zip(sources, destinations)):
        edge_dict[source].append(dest)
    return edge_dict


def get_stacked_tensors_from_dict(dictionary: dict, ids):
    return torch.stack([dictionary[id] for id in ids])
