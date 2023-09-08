import argparse


def get_data_config_parser():
    parser = argparse.ArgumentParser(description="Data processing.")

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset", 
        type=str, 
        default="imagenet", 
        help="Dataset name(default: %(default)s)."
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--hdf5_dir", 
        type=str, 
        help="HDF5 directory(default: %(default)s)."
    )

    return parser

def get_config_parser():
    parser = argparse.ArgumentParser(description="Run experiments.")

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--batch_size", type=int, default=32, help="batch size (default: %(default)s)."
    )
    data.add_argument(
        "--scaling_factor", 
        type=int, 
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        default=6, 
        help="Scaling factor(default: %(default)s)."
    )

    data.add_argument(
        "--data_aug", 
        dest="data_aug", 
        action="store_true",
        help="Adding data augmentation"
    )

    data.add_argument(
        "--no-data_aug", 
        dest="data_aug", 
        action="store_false",
        help="Without data augmentation"
    )
    data.set_defaults(feature=False)
    
    data.add_argument(
        "--times", 
        type=int, 
        choices=[0, 1, 2],
        default=1, 
        help="How many times to augment the data"
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
    model.add_argument(
        "--model_type",
        type=str,
        default='resnet',
        choices=['resnet', 'vit'],
        help="Architecture selected."
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

    optimization = parser.add_argument_group("Checkpoints")
    optimization.add_argument(
        "--load_checkpoint",
        dest="load_checkpoint", 
        action="store_true",
        help="If we will load a saved checkpoint and continue training.",
    )

    optimization.add_argument(
        "--no-load_checkpoint", 
        dest="load_checkpoint", 
        action="store_false",
        help="Without a checkpoint."
    )

    optimization.set_defaults(feature=False)

    optimization.add_argument(
        "--save_checkpoint",
        dest="save_checkpoint", 
        action="store_true",
        help="Save a checkpoint at every epoch.",
    )

    optimization.add_argument(
        "--no-save_checkpoint", 
        dest="save_checkpoint", 
        action="store_false",
        help="Don't save checkpoints."
    )

    optimization.set_defaults(feature=False)

    optimization.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoint.pth",
        help="checkpoint file path (default: %(default)s).",
    )

    exp = parser.add_argument_group("Experiment config")
    exp.add_argument(
        "--logdir",
        type=str,
        default='exps/',
        help="experiment result's folder.",
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
