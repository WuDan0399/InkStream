from GIN import GIN
from inkstream import inkstream
from load_dataset import load_dataset
from EventQueue import *
from utils import *

class inkstream_gin(inkstream):
    def __init__(
        self,
        model,
        intr_result_dir,
        aggregator: str = "max",
        verify: bool = False,
        verification_tolerance: float = 1e-5,
        out_channels: int = 1,
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
            [self.aggregator, "user_apply", lambda x: self.model.convs[0].nn(x).relu()],
            [self.aggregator, "user_apply", lambda x: self.model.convs[1].nn(x).relu()],
            [self.aggregator, "user_apply", lambda x: self.model.convs[2].nn(x).relu()],
            [self.aggregator, "user_apply", lambda x: self.model.convs[3].nn(x).relu()],
            [self.aggregator, "user_apply", lambda x: self.model.convs[4].nn(x).relu()],
        ]

    def user_apply(
        self,
        events: dict,
        base_value: torch.Tensor,
        intm_initial: dict = None,
        it_layer: int = 0,
        node: int = -1,
    ):
        if (
            "user" not in events.keys()
        ): 
            right_side = intm_initial[f"layer{it_layer+1}"]["before"][node].to(
                device)
        else:
            assert len(events["user"]) == 1
            right_side = events["user"][0].to(device)
        return base_value + right_side

    def user_reducer(self, messages: list):
        return messages

    def user_propagate(self, node: int, value: torch.Tensor, event_queue: EventQueue):
        event_queue.push_user_event("user", node, value)

    def inc_aggregator_pair(self, message_a, message_b): 
        return torch.maximum(message_a, message_b)

    def inc_aggregator(self, message_list: torch.Tensor):
        return torch.max(message_list, dim=0)

    def monotonic_aggregator(self, messages: list):
        if len(messages) == 2:
            return torch.maximum(messages[0], messages[1])
        else:
            return torch.max(torch.stack(messages), dim=0).values


def main():
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    data = dataset[0]
    batch_size = int(args.perbatch)
    if args.dataset == 'papers':
        model = GIN(dataset.num_features, dataset.num_classes + 1, args).to(device)
        out_channels = dataset.num_classes+1
    else:
        model = GIN(dataset.num_features, dataset.num_classes, args).to(device)
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

    conditions_dir = osp.join("examples", "condition_distribution", "GIN")
    create_directory(conditions_dir)
    time_dir = osp.join("examples", "timing_result", "incremental")
    create_directory(time_dir)

    starter = inkstream_gin(
        model,
        intr_result_dir,
        aggregator=args.aggr,
        verify=False,
        out_channels=out_channels
    )
    _, exec_time_dist = starter.batch_incremental_inference(data)
    print("Execution time (s) of InkStream for each sample is:")
    print(exec_time_dist)


if __name__ == "__main__":
    main()

