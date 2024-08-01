from inkstream import inkstream
from utils import *
from GCN import GCN

from load_dataset import load_dataset

class inkstream_gcn(inkstream):
    def __init__(
        self,
        model,
        intr_result_dir,
        aggregator: str = "min",
        verify: bool = False,
        verification_tolerance: float = 1e-5,
        out_channels:int = 1,
    ):
        super().__init__(
            model,
            intr_result_dir,
            aggregator,
            verify,
            verification_tolerance,
            out_channels,
        )
        self.model_config = [
            [self.aggregator, self.conv1_bias, self.conv2],
            [self.aggregator, self.conv2_bias],
        ]

    def conv1(self, x):
        return self.model.conv1.lin(x)

    def conv2(self, x):
        return self.model.conv2.lin(x)

    def conv1_bias(self, x):
        return (x + self.model.conv1.bias).relu()

    def conv2_bias(self, x):
        return x + self.model.conv2.bias


def main():
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    data = dataset[0]
    batch_size = int(args.perbatch)
    if args.dataset == "papers":
        model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes+1, args).to(device)
        out_channels = dataset.num_classes+1
    else:
        model = GCN(
            dataset.num_features, args.hidden_channels, dataset.num_classes, args
        ).to(device)
        out_channels = dataset.num_classes
    model = load_available_model(model, args)

    intr_result_dir = osp.join(
        "examples",
        "intermediate",
        args.dataset,
        "min", 
        args.stream,
        f"batch_size_{batch_size}",
    )

    conditions_dir = osp.join("examples", "condition_distribution", "GCN")
    create_directory(conditions_dir)
    time_dir = osp.join("examples", "timing_result", "incremental")
    create_directory(time_dir)

    starter = inkstream_gcn(model, intr_result_dir, aggregator=args.aggr, verify=False, out_channels=out_channels)
    _, exec_time_dist = starter.batch_incremental_inference(data)
    print("Execution time of InkStream for each sample is:")
    print(exec_time_dist)



if __name__ == "__main__":
    main()
