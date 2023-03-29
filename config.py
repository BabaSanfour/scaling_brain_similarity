import argparse

def get_config_parser():
    parser = argparse.ArgumentParser(description="Run experiments.")

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--batch_size", type=int, default=32, help="batch size (default: %(default)s)."
    )
    data.add_argument(
        "--scaling_factor", 
        type=int, 
        choices=[0, 1, 2, 3, 4, 5, 6],
        default=6, 
        help="Scaling factor(default: %(default)s)."
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model_config",
        type=str,
        default='./model_configs/resnet50.json',
        help="Path to model config json file"
    )

    model.add_argument(
        "--model_name",
        type=str,
        default='resnet.pth',
        help="Name of the model"
    )

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of epochs for training (default: %(default)s).",
    )
    optimization.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "momentum", "adam", "adamw"],
        help="choice of optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="learning rate optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum for SGD optimizer (default: %(default)s).",
    )
    optimization.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="weight decay (default: %(default)s).",
    )

    exp = parser.add_argument_group("Experiment config")
    exp.add_argument(
        "--logdir",
        type=str,
        default='exps/',
        help="unique experiment identifier (default: %(default)s).",
    )
    exp.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for repeatability (default: %(default)s).",
    )

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to store tensors on (default: %(default)s).",
    )
    return parser
