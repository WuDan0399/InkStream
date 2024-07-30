import torch
from tqdm import tqdm
from utils import *
from load_dataset import load_dataset
from GCN import GCN
import pickle


def select_edges_by_degree_with_masks(edge_list, num_indices=100, top_k=10, highest=True):
    """
    Select edges connected to the top_k vertices with the highest or lowest degrees and return masks for manipulation.

    Parameters:
    - edge_list: Tensor of shape (2, num_edges) representing the edge list of the graph.
    - num_indices: The total number of edge indices to be divided between add and delete masks.
    - top_k: The number of vertices to consider based on their degree.
    - highest: If True, select vertices with the highest degrees; otherwise, select those with the lowest degrees.

    Returns:
    - add_mask: Mask for the first half of randomly selected edges.
    - delete_mask: Mask for the second half of randomly selected edges.
    - before_mask: Mask for all edges not included in add_mask.
    - after_mask: Mask for all edges not included in delete_mask.
    """
    # Calculate the degree of each vertex
    all_vertices = edge_list.flatten()
    unique_vertices, counts = torch.unique(all_vertices, return_counts=True)

    # Find the top_k vertices based on degree
    if highest:
        _, top_vertex_indices = torch.topk(counts, top_k)
    else:
        _, top_vertex_indices = torch.topk(counts, top_k, largest=False)
    top_vertices = unique_vertices[top_vertex_indices]

    # Select edges connected to the top_k vertices
    mask = torch.any(torch.isin(edge_list, top_vertices).T, dim=1)
    selected_edges_indices = torch.nonzero(mask).flatten()

    # Shuffle and split the selected indices into add and delete groups
    shuffled_indices = selected_edges_indices[torch.randperm(len(selected_edges_indices))]
    split_point = int(num_indices // 2)
    add_indices = shuffled_indices[:split_point]
    delete_indices = shuffled_indices[split_point:split_point * 2]  # Ensure this doesn't exceed num_indices

    # Create masks based on the selected indices
    add_mask = torch.zeros(len(mask), dtype=torch.bool)
    add_mask[add_indices] = True

    delete_mask = torch.zeros(len(mask), dtype=torch.bool)
    delete_mask[delete_indices] = True

    # Masks for all edges not included in add and delete masks
    before_mask = ~add_mask
    after_mask = ~delete_mask

    return add_mask, delete_mask, before_mask, after_mask


def select_edges_by_degree_with_direction_masks(edge_list, num_indices:int=100, top_k=10, highest=True, edge_type='mixed'):
    """
    Select edges connected to the top_k vertices with the highest or lowest degrees and return masks for manipulation.
    Allows selection based on edge direction relative to high-degree vertices.

    Parameters:
    - edge_list: Tensor of shape (2, num_edges) representing the edge list of the graph.
    - num_indices: The total number of edge indices to be divided between add and delete masks.
    - top_k: The number of vertices to consider based on their degree.
    - highest: If True, select vertices with the highest degrees; otherwise, select those with the lowest degrees.
    - edge_type: 'incoming', 'outgoing', or 'mixed' edges relative to the top_k vertices.

    Returns:
    - add_mask, delete_mask, before_mask, after_mask: Masks for edge manipulation.
    """
    # Calculate the degree of each vertex based on edge_type
    if edge_type in ['incoming', 'mixed']:
        vertex_positions = [1] if edge_type == 'incoming' else [0, 1]
        vertices_to_consider = torch.cat([edge_list[pos] for pos in vertex_positions])
    else:  # Outgoing
        vertices_to_consider = edge_list[0]

    unique_vertices, counts = torch.unique(vertices_to_consider, return_counts=True)

    # Select the top_k vertices based on degree
    if highest:
        _, top_vertex_indices = torch.topk(counts, top_k)
    else:
        _, top_vertex_indices = torch.topk(counts, top_k, largest=False)
    top_vertices = unique_vertices[top_vertex_indices]

    # Define a function to test if an edge is connected to top_vertices based on edge_type
    def is_edge_connected(edge):
        if edge_type == 'incoming':
            return torch.isin(edge[1], top_vertices)
        elif edge_type == 'outgoing':
            return torch.isin(edge[0], top_vertices)
        return torch.isin(edge, top_vertices).any()

    # Apply the function to select edges
    mask = torch.tensor([is_edge_connected(edge) for edge in edge_list.T])
    selected_edges_indices = torch.nonzero(mask).flatten()

    # Shuffle and split the selected indices into add and delete groups
    shuffled_indices = selected_edges_indices[torch.randperm(len(selected_edges_indices))]
    split_point = int(num_indices // 2)
    add_indices = shuffled_indices[:split_point]
    delete_indices = shuffled_indices[split_point:split_point * 2]  # Ensure this doesn't exceed num_indices

    # Create masks based on the selected indices
    add_mask = torch.zeros(len(mask), dtype=torch.bool)
    add_mask[add_indices] = True

    delete_mask = torch.zeros(len(mask), dtype=torch.bool)
    delete_mask[delete_indices] = True

    # Masks for edges not included in add and delete masks
    before_mask = ~add_mask
    after_mask = ~delete_mask

    return add_mask, delete_mask, before_mask, after_mask


def get_direct_affected_tensor(inserted_edges: torch.tensor, removed_edges: torch.tensor) -> set:
    dest_add = torch.unique(inserted_edges[1, :])
    dest_rm = torch.unique(removed_edges[1, :])
    all_unique = torch.unique(torch.cat((dest_add, dest_rm)))
    direct_affected_nodes = set(all_unique.tolist())
    return direct_affected_nodes


@torch.no_grad()
def inference_for_intermediate_result(model, loader):
    model.eval()
    intermediate_result_each_layer = defaultdict(lambda: defaultdict(lambda: torch.empty((0))))
    for batch in tqdm(loader):
        batch = batch.to(device)
        batch_size = batch.batch_size
        _, _, batch_intermediate_result_per_layer = model(batch.x, batch.edge_index)
        for layer in batch_intermediate_result_per_layer:
            if len(intermediate_result_each_layer[layer]['a-']) != 0:
                intermediate_result_each_layer[layer]['a-'] = torch.concat((intermediate_result_each_layer[layer]["a-"],
                                                                            batch_intermediate_result_per_layer[layer][
                                                                                "a-"][:batch_size].cpu()))
            else:
                intermediate_result_each_layer[layer]['a-'] = batch_intermediate_result_per_layer[layer]["a-"][
                                                              :batch_size].cpu()

            if len(intermediate_result_each_layer[layer]['a']) != 0:
                intermediate_result_each_layer[layer]['a'] = torch.concat((intermediate_result_each_layer[layer]["a"],
                                                                           batch_intermediate_result_per_layer[layer][
                                                                               "a"][:batch_size].cpu()))
            else:
                intermediate_result_each_layer[layer]['a'] = batch_intermediate_result_per_layer[layer]["a"][
                                                             :batch_size].cpu()
    return intermediate_result_each_layer


def intm_affected(model, data, edges, nlayer: int = 2, inserted_edges=None, removed_edges=None, init_in_edge_dict=None,
                  final_in_edge_dict=None, init_out_edge_dict=None, final_out_edge_dict=None) -> Tuple[dict, set]:
    if isinstance(inserted_edges, torch.Tensor): 
        direct_affected_nodes = get_direct_affected_tensor(inserted_edges, removed_edges)
    else:
        direct_affected_nodes = set([dst for _, dst in inserted_edges + removed_edges])
    dicts = [i for i in [init_out_edge_dict, init_in_edge_dict, final_out_edge_dict, final_in_edge_dict] if i != None]
    total_affected_nodes = affected_nodes_each_layer(dicts, direct_affected_nodes, depth=nlayer - 1)
    affected_nodes = torch.LongTensor(list(total_affected_nodes[nlayer - 1]))
    print(f"total affected nodes: {len(affected_nodes)}")

    data.edge_index = edges
    loader = data_loader(data, num_layers=nlayer, num_neighbour_per_layer=-1, separate=False,
                         input_nodes=affected_nodes)
    intm_raw = inference_for_intermediate_result(model, loader)
    intm = {
        it_layer: {
            "before": {
                affected_nodes[i].item(): value["a-"][i]
                for i in range(len(affected_nodes))
            }, "after": {
                affected_nodes[i].item(): value["a"][i]
                for i in range(len(affected_nodes))
            }, }
        for it_layer, value in intm_raw.items()
    }
    return intm, total_affected_nodes[nlayer - 1] 


if __name__ == '__main__':
    depth = 2
    use_sampled_graph = False

    create_directory(osp.join("examples", "theoretical"))

    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    batch_size = int(args.perbatch)
    batch_sizes = defaultConfigs.batch_sizes
    num_samples = defaultConfigs.num_samples
    num_sample = num_samples[batch_sizes.index(batch_size)] if batch_size in batch_sizes else None
    num_sample = 5

    data = dataset[0]
    f = open(osp.join("examples", "theoretical", f"2layerGCN_affected_vs_real_{args.dataset}_{batch_size}.txt"), "w")
    f.write(
        f"batch_size\t#layer\texample_id\tth_affected_nodes(sampled)\treal_affected_nodes(sampled)\n")

    if args.dataset == 'papers':
        model = GCN(dataset.num_features, 256, dataset.num_classes + 1, args).to(device)
    else:
        model = GCN(dataset.num_features, 256, dataset.num_classes, args).to(device)

    available_model = []
    name_prefix = f"{args.dataset}_GCN_{args.aggr}"
    for file in os.listdir("examples/trained_model"):
        if re.match(name_prefix + "_[0-9]+_[0-9]\.[0-9]+\.pt", file):
            available_model.append(file)
    if len(available_model) == 0:
        print("no trained model")
    else:
        model = load(model, available_model[0])

    folder = osp.join("examples", "intermediate", args.dataset, args.aggr, args.stream,
                      f"batch_size_{batch_size}")
    entries = os.listdir(folder)
    data_folders = [entry for entry in entries if entry.isdigit() and os.path.isdir(os.path.join(folder, entry))][
                   :num_sample]

    pbar = tqdm(data_folders)
    total_all_cases = []
    change_all_cases = []
    case_id = -1
    if use_sampled_graph:
        print("Use saved cases for sampled graphs")
    else:
        print("Use snapshots with randomly removed edges of full graph")
        all_edges = data.edge_index
        edge_dict_file_name = f'{args.dataset}_full_dict.pickle'
        if os.path.exists(edge_dict_file_name):
            with open(edge_dict_file_name, 'rb') as file:
                edge_dict_full = pickle.load(file)
        else:
            edge_dict_full = to_dict(data.edge_index)
            with open(edge_dict_file_name, 'wb') as file:
                pickle.dump(edge_dict_full, file)

    for data_dir in data_folders[:100]:
        try:
            if use_sampled_graph:
                case_id += 1
                initial_edges = torch.load(osp.join(folder, data_dir, "initial_edges.pt")) 
                final_edges = torch.load(osp.join(folder, data_dir, "final_edges.pt"))
                inserted_edges, removed_edges = [], []
                if osp.exists(osp.join(folder, data_dir, "inserted_edges.pt")):
                    inserted_edges = torch.load(osp.join(folder, data_dir, "inserted_edges.pt"))
                    inserted_edges = [(src.item(), dst.item()) for src, dst in
                                      zip(inserted_edges[0], inserted_edges[1])]
                if osp.exists(osp.join(folder, data_dir, "removed_edges.pt")):
                    removed_edges = torch.load(osp.join(folder, data_dir, "removed_edges.pt"))
                    removed_edges = [(src.item(), dst.item()) for src, dst in zip(removed_edges[0], removed_edges[1])]
                edge_dict_file_name = f'{args.dataset}_{data_dir}_dict.pickle'
                if os.path.exists(edge_dict_file_name):
                    with open(edge_dict_file_name, 'rb') as file:
                        init_out_edge_dict_sampled = pickle.load(file)
                else:
                    init_out_edge_dict_sampled = to_dict(initial_edges)
                    with open(edge_dict_file_name, 'wb') as file:
                        pickle.dump(init_out_edge_dict_sampled, file)
                final_edge_dict_file_name = f'{args.dataset}_{data_dir}_final_dict.pickle'
                if os.path.exists(final_edge_dict_file_name):
                    with open(final_edge_dict_file_name, 'rb') as file:
                        final_out_edge_dict_sampled = pickle.load(file)
                else:
                    final_out_edge_dict_sampled = to_dict(final_edges)
                    with open(final_edge_dict_file_name, 'wb') as file:
                        pickle.dump(final_out_edge_dict_sampled, file)
                direct_affected_nodes = set([dst for _, dst in inserted_edges + removed_edges])
                intm_before, affected_before = intm_affected(model, data, initial_edges, 2, inserted_edges,
                                                             removed_edges,
                                                             init_out_edge_dict_sampled, final_out_edge_dict_sampled)
                del initial_edges
                intm_after, affected_after = intm_affected(model, data, final_edges, 2, inserted_edges, removed_edges,
                                                           init_out_edge_dict_sampled, final_out_edge_dict_sampled)
                del final_edges
                del init_out_edge_dict_sampled
                torch.cuda.empty_cache()

            else:
                # Random select edges
                # indices = torch.randperm(all_edges.size(1))
                # edges_before = all_edges[:, indices[:-batch_size // 2]]
                # inserted_edges = all_edges[:, indices[-batch_size // 2:]]
                # removed_edges = all_edges[:, indices[:batch_size // 2]]
                # edges_after = all_edges[:, indices[batch_size // 2:]]

                # Select edges of high-(low-)deg nodes
                add_mask, delete_mask, before_mask, after_mask = select_edges_by_degree_with_masks(all_edges, args.perbatch, top_k=1000, highest=False)
                #add_mask, delete_mask, before_mask, after_mask = select_edges_by_degree_with_direction_masks(all_edges,
                #                                                                                            args.perbatch,
                #                                                                                             top_k=1000,
                #                                                                                             highest=False,
                #                                                                                             edge_type="mixed")
                edges_before = all_edges[:, before_mask]
                inserted_edges = all_edges[:, add_mask]
                edges_after = all_edges[:, after_mask]
                removed_edges = all_edges[:, delete_mask]

                direct_affected_nodes = get_direct_affected_tensor(inserted_edges, removed_edges)
                intm_before, affected_before = intm_affected(model, data, edges_before, 2, inserted_edges,
                                                             removed_edges, edge_dict_full)
                intm_after, affected_after = intm_affected(model, data, edges_after, 2, inserted_edges, removed_edges,
                                                           edge_dict_full)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        total_affected = affected_before | affected_after
        total_nodes = len(total_affected)  
        changed = 0
        for node in total_affected:
            if not torch.all(torch.isclose(intm_before[f"layer{depth}"]["after"][node],
                                           intm_after[f"layer{depth}"]["after"][node])):
                changed += 1
        print(f"Total/changed: {total_nodes}/{changed}")
        total_all_cases.append(total_nodes)
        change_all_cases.append(changed)
        f.write(
            f"{batch_size}\t{depth}\t{case_id}\t{total_nodes}\t{changed}\n")
    total_all_cases = np.array(total_all_cases)
    change_all_cases = np.array(change_all_cases)
    f.write(
        f"average real:theoretical = {np.mean(change_all_cases / total_all_cases)}\n")
