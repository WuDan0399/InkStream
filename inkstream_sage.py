from inkstream import inkstream
from utils import *
from SAGE import SAGE
from EventQueue import *
from load_dataset import load_dataset

class inkstream_sage(inkstream):
    def __init__(
        self,
        model,
        intr_result_dir,
        aggregator: str = "min",
        verify: bool = False,
        verification_tolerance: float = 1e-5,
        out_channels: int = 1,
        ego_net: bool = False,
        multi_thread: int = 0
    ):
        super().__init__(
            model,
            intr_result_dir,
            aggregator,
            verify,
            verification_tolerance,
            out_channels,
            ego_net,
            multi_thread,
        )
        self.model_config = [
            [
                self.aggregator,
                lambda x: self.model.convs[0].lin_l(x),
                "user_apply",
                lambda x: x.relu(),
            ],
            [
                self.aggregator,
                lambda x: self.model.convs[1].lin_l(x),
                "user_apply",
                lambda x: x.relu(),
            ],
        ]

    def layer1(self, x):
        return self.model.convs[0].lin_l(x)

    def layer2(self, x):
        return self.model.convs[1].lin_l(x)

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
        return eval(f"base_value + self.model.convs[{it_layer}].lin_r(right_side)")

    def user_reducer(self, messages: list):
        return messages

    def user_propagate(self, node: int, value: torch.Tensor, event_queue: EventQueue):
        event_queue.push_user_event("user", node, value)


def main():
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    data = dataset[0]
    batch_size = int(args.perbatch)
    if args.dataset == 'papers':
        model = SAGE(dataset.num_features, 256, dataset.num_classes + 1, args).to(device)
        out_channels = dataset.num_classes + 1
    else:
        model = SAGE(dataset.num_features, 256, dataset.num_classes, args).to(device)
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

    time_dir = osp.join("examples", "timing_result", "incremental")
    create_directory(time_dir)
    conditions_dir = osp.join("examples", "condition_distribution", "SAGE")
    create_directory(conditions_dir)

    starter = inkstream_sage(model, intr_result_dir,
                             multi_thread=args.mt, aggregator=args.aggr, verify=False, out_channels=out_channels)
    _, exec_time_dist = starter.batch_incremental_inference(data)
    print("Execution time of InkStream for each sample is:")
    print(exec_time_dist)


if __name__ == "__main__":
    main()
