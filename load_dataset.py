from torch_geometric.datasets import Reddit, Planetoid, CitationFull, Yelp, AmazonProducts
from utils import *

def generate_snapshots(data:Data, dataset_name:str, stream:str, args:argparse.Namespace):
    batch_sizes = defaultConfigs.batch_sizes
    num_samples = defaultConfigs.num_samples
    timing_sampler(data, args)
    exist = True
    for batch_size in batch_sizes:
        if not osp.exists(osp.join(root, "dynamic", "examples", "intermediate", dataset_name, "min", "add",
                                   f"batch_size_{batch_size}", str(0))):
            exist = False
            break
    if exist:
        return

    for batch_size, num_sample in zip(batch_sizes, num_samples):
        if osp.exists(osp.join(root, "dynamic", "examples", "intermediate", dataset_name, "min", stream,
                                   f"batch_size_{batch_size}", str(0))):
            continue

        print(f"Generating graph topology snapshots for batch size {batch_size}, sample {num_sample}.")
        for i in range(num_sample):
            out_folder = osp.join(root, "dynamic", "examples", "intermediate", dataset_name, "min", stream,
                              f"batch_size_{batch_size}", str(i))
            create_directory(out_folder)
            # edge selection
            initial_edges, final_edges, inserted_edges, removed_edges = get_graph_dynamics(data.edge_index,
                                                                                           batch_size,
                                                                                           stream)
            torch.save(initial_edges, (osp.join(out_folder, "initial_edges.pt")))
            torch.save(final_edges, (osp.join(out_folder, "final_edges.pt")))
            if inserted_edges.shape[1]:
                torch.save(inserted_edges, (osp.join(out_folder, "inserted_edges.pt")))
            if removed_edges.shape[1]:
                torch.save(removed_edges, (osp.join(out_folder, "removed_edges.pt")))
    print("generate snapshots, end.")


def load_dataset(args: argparse.Namespace):
    from ogb.nodeproppred import PygNodePropPredDataset
    if args.dataset == "Cora":
        dataset = Planetoid(osp.join(root, "datasets", "Planetoid"), "Cora")
    elif args.dataset == "PubMed":
        dataset = Planetoid(osp.join(root, "datasets", "Planetoid"), "PubMed")
    elif args.dataset == "reddit":
        dataset = Reddit(osp.join(root, "datasets", "Reddit"))
    elif args.dataset == "cora":
        dataset = CitationFull(
            osp.join(root, "datasets", "CitationFull"), "Cora")
    elif args.dataset == "yelp":
        dataset = Yelp(osp.join(root, "datasets", "Yelp"))
    elif args.dataset == "products":
        dataset = PygNodePropPredDataset(name="ogbn-products", root=osp.join(root, "datasets"))
    elif args.dataset == "papers":
        dataset = PygNodePropPredDataset(name="ogbn-papers100M", root=osp.join(root, "datasets"))
    else:
        print("No such dataset. Available: Cora/cora/PubMed/reddit/yelp/products/papers")

    generate_snapshots(dataset[0], args.dataset, args.stream, args)  # generate snapshots for the first(only) graph

    return dataset