import os
import json
import time
import wandb
import warnings

from resnet import resnet
from ViT import ViT
from utils.data_loader import dataloader
from utils.config import get_config_parser
from utils.utils import seed_experiment, get_model_size

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

import logging

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        dataset_sizes: dict,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss._Loss,
        starting_epoch: int,
        max_epochs: int,
        save_every: int,
        wandb: wandb,
        gpu_id: int,
    ) -> None:
        """
        Initialize the Trainer object for model training.

        Args:
            model (nn.Module): The neural network model to train.
            data_loader (torch.utils.data.DataLoader): Data loader for training, validation, and testing datasets.
            dataset_sizes (dict): Dictionary containing sizes of training, validation, and test datasets.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters during training.
            criterion (torch.nn.modules.loss._Loss): Loss function used for training.
            starting_epoch (int): The epoch to start training from.
            max_epochs (int): Maximum number of epochs to train the model.
            save_every (int): Frequency at which to save model checkpoints (every n epochs).
            wandb (wandb): Wandb object for logging training metrics.
            gpu_id (int): GPU ID where the model will be trained.

        Returns:
            None
        """
        self.model = model
        self.train_data = data_loader['train']
        self.valid_data = data_loader['valid']
        self.test_data = data_loader['test']
        self.train_size = dataset_sizes["train"]
        self.valid_size = dataset_sizes["valid"]
        self.test_size = dataset_sizes["test"]
        self.optimizer = optimizer
        self.criterion = criterion
        self.starting_epoch = starting_epoch
        self.max_epochs = max_epochs
        self.save_every = save_every
        self.wandb = wandb
        self.gpu_id = gpu_id
    
    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> tuple:
        """
        Run a single batch through the model and return loss and accuracy.

        Args:
            source (torch.Tensor): Input tensor.
            targets (torch.Tensor): Target tensor.

        Returns:
            tuple: A tuple containing the following values:
                - loss (float): Loss value.
                - n_correct (int): Number of correct predictions.
                - outputs (torch.Tensor): Model outputs.
        """
        # Zero the gradients to avoid accumulation
        self.optimizer.zero_grad()

        # Forward pass through the model
        outputs = self.model(source)

        # Compute the loss using the specified criterion
        loss = self.criterion(outputs, targets)

        # Backward pass and optimization step
        loss.backward()
        self.optimizer.step()

        # Compute the number of correct predictions
        n_correct = (torch.max(outputs, 1)[1].view(targets.data.size()) == targets.data).sum().item()

        return loss.item(), n_correct, outputs


    def _run_epoch(self, epoch: int) -> None:
        """
        Run a single training epoch, updating the model's parameters.

        Args:
            epoch (int): The current epoch number.

        Returns:
            None
        """
        # Set the epoch for the data sampler
        self.train_data.sampler.set_epoch(epoch)

        # Get the batch size from the first batch in the training data
        batch_size = len(next(iter(self.train_data))[0])

        # Initialize counters and lists
        n_corrects = 0
        running_loss = 0.0
        all_outputs = []
        all_labels = []

        # Print information about the training process
        logger.info(f"GPU {self.gpu_id} - Epoch {epoch} - Batch size {batch_size} - steps {self.train_size // batch_size} - Total steps {len(self.train_data)}")

        # Enable gradient computation
        with torch.set_grad_enabled(True):
            # Set the model to training mode
            self.model.train()

            # Iterate through batches in the training data
            start_time = time.time()
            for source, targets in self.train_data:
                pass
                # Move data to the GPU
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)

                # Run a single batch through the model and obtain loss and accuracy
                loss, n_correct, outputs = self._run_batch(source, targets)

                # Update counters and lists
                n_corrects += n_correct
                running_loss += loss * batch_size
                all_outputs.append(outputs)
                all_labels.append(targets)
            end_time = time.time()
            loading_time = end_time - start_time

            # Print or log the loading time
            print(f"Data loading time: {loading_time:.2f} seconds")

            # Reduce and aggregate values across all GPUs
            total_correct = torch.tensor(n_corrects, dtype=torch.long, device=self.gpu_id)
            total_loss = torch.tensor(running_loss, dtype=torch.float, device=self.gpu_id)

            dist.reduce(total_correct, op=dist.ReduceOp.SUM, dst=0)
            dist.reduce(total_loss, op=dist.ReduceOp.SUM, dst=0)

            # Calculate average training accuracy and loss on the master GPU
            avg_train_acc = 0
            avg_train_loss = 0
            if dist.get_rank() == 0:
                avg_train_acc = 100. * total_correct.item() / self.train_size
                avg_train_loss = total_loss.item() / self.train_size
                logger.info(f"GPU {self.gpu_id} - Train Accuracy: {avg_train_acc:.3f} - Train Loss: {avg_train_loss:.3f}")

            return avg_train_loss, avg_train_acc

    def _run_validation(self, epoch: int) -> None:
        """
        Run validation after each training epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            None
        """
        # Disable gradient computation during validation
        with torch.no_grad():
            # Set the model to evaluation mode
            self.model.eval()

            # Initialize counters and lists
            n_corrects = 0
            running_loss = 0.0
            all_outputs = []
            all_labels = []

            # Set the epoch for the data sampler
            self.valid_data.sampler.set_epoch(epoch)

            # Iterate through batches in the validation data
            for source, targets in self.valid_data:
                # Move data to the GPU
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)

                # Run a forward pass through the model
                outputs = self.model(source)

                # Compute the number of correct predictions
                n_corrects += (torch.max(outputs, 1)[1].view(targets.data.size()) == targets.data).sum().item()

                # Compute the loss
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * source.size(0)

                # Append outputs and labels to lists
                all_outputs.append(outputs)
                all_labels.append(targets)

            # Reduce and aggregate values across all GPUs
            total_correct = torch.tensor(n_corrects, dtype=torch.long, device=self.gpu_id)
            total_loss = torch.tensor(running_loss, dtype=torch.float, device=self.gpu_id)

            dist.reduce(total_correct, op=dist.ReduceOp.SUM, dst=0)
            dist.reduce(total_loss, op=dist.ReduceOp.SUM, dst=0)

            # Calculate average validation accuracy and loss on the master GPU
            avg_valid_acc = 0
            avg_valid_loss = 0
            if dist.get_rank() == 0:
                avg_valid_acc = 100. * total_correct.item() / self.valid_size
                avg_valid_loss = total_loss.item() / self.valid_size
                logger.info(f"GPU {self.gpu_id} - Validation Accuracy: {avg_valid_acc:.3f} - Validation Loss: {avg_valid_loss:.3f}")

            return avg_valid_loss, avg_valid_acc


    def _save_checkpoints(self, epoch: int) -> None:
        """
        Save checkpoints during the training process.

        Args:
            epoch (int): The current epoch number.

        Returns:
            None
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        checkpoint_filename = f"checkpoint_{epoch}.pth"
        torch.save(checkpoint, checkpoint_filename)
        logger.info(f"GPU {self.gpu_id} - Saved checkpoint: {checkpoint_filename}")


    def train(self) -> None:
        """
        Train the model.

        Args:
            None

        Returns:
            None
        """
        best_acc = 0.0
        since = time.time()

        for epoch in range(self.starting_epoch, self.max_epochs):
            # Run training and validation for the current epoch
            train_loss, train_acc = self._run_epoch(epoch)
            valid_loss, valid_acc = self._run_validation(epoch)

            # Log metrics to WandB for GPU 0 only (to avoid duplicate logging)
            if self.gpu_id == 0:
                self.wandb.log({
                    "Train Loss": train_loss,
                    "Train Accuracy": train_acc,
                    "Validation Loss": valid_loss,
                    "Validation Accuracy": valid_acc
                })

                # Save checkpoints every 'save_every' epochs
                if epoch % self.save_every == 0:
                    self._save_checkpoints(epoch)

                # Update best accuracy and save best model
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model = self.model.module
        torch.cuda.empty_cache()

        # Finalize training and log results for GPU 0 only
        if self.gpu_id == 0:
            time_elapsed = time.time() - since
            logger.info(f'Training complete in {time_elapsed // 3600} h {(time_elapsed % 3600) // 60} m {time_elapsed % 60} s')
            logger.info(f'BEST VALID ACC: {best_acc:.3f}')
            self.model.module = best_model
            self._save_checkpoints("best")


    def test(self):
        """
        Test the model.

        Returns:
            None
        """
        # Set the model to evaluation mode
        self.model.eval()

        n_dev_correct = 0
        test_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        # Set the epoch for the test data sampler
        self.test_data.sampler.set_epoch(0)

        with torch.no_grad():
            # Iterate over test data
            for source, targets in self.test_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                outputs = self.model(source)

                # Compute test accuracy and loss
                n_dev_correct += (torch.max(outputs, 1)[1].view(targets.data.size()) == targets.data).sum().item()
                loss = criterion(outputs, targets)
                test_loss += loss.item() * source.size(0)

        # Reduce and sum metrics across distributed training
        total_correct = torch.tensor(n_dev_correct, dtype=torch.long, device=self.gpu_id)
        total_loss = torch.tensor(test_loss, dtype=torch.float, device=self.gpu_id)
        dist.reduce(total_correct, op=dist.ReduceOp.SUM, dst=0)
        dist.reduce(total_loss, op=dist.ReduceOp.SUM, dst=0)

        # Calculate average test accuracy and loss for GPU 0 only
        avg_test_acc = 100. * total_correct.item() / self.test_size
        avg_test_loss = total_loss.item() / self.test_size

        if dist.get_rank() == 0:
            # Print and log the results for GPU 0 only
            logger.info(f'TEST LOSS: {avg_test_loss:.3f}')
            logger.info(f'TEST Accuracy (Top1): {avg_test_acc:.3f}')
            self.wandb.log({
                "Test Loss": avg_test_loss, 
                "Test Accuracy": avg_test_acc
            })
        torch.cuda.empty_cache()


def load_train_objects(args, rank) -> tuple:
    """
    Load the model, optimizer, loss function, and data loader.

    Args:
        args (argparse.ArgumentParser): Command line arguments.
    
    Returns:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Model optimizer.
        criterion (torch.nn.modules.loss._Loss): Loss function.
        dataset_loader (torch.utils.data.DataLoader): Data loader.
        dataset_sizes (dict): Dataset sizes.
        starting_epoch (int): Starting epoch for training.
        wandb (wandb): Wandb object for logging (optional).
        rank (int): Process rank within the distributed environment.
    """
    # Load dataset
    if args.random_seed != 42:
        logger.info(f"Using random seed {args.random_seed}")
        dataset_loader, dataset_sizes = dataloader(args.batch_size, f"{args.scaling_factor}_{args.random_seed}", args.data_aug, args.times)
    else:
        dataset_loader, dataset_sizes = dataloader(args.batch_size, args.scaling_factor, args.data_aug, args.times)


    # Load model configuration
    if args.model_config is not None:
        logger.info(f'Loading model config from {args.model_config}')
        with open(args.model_config) as f:
            model_config = json.load(f)
    else:
        raise ValueError('Please provide a model config json')

    # Add the number of classes to the model configuration
    model_config['num_classes'] = args.num_classes

    # Initialize the model based on the model type
    if args.model_type == "resnet":
        model = resnet(**model_config)
    else:
        model = ViT(**model_config)

    # Initialize the optimizer based on the selected option
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "momentum":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    # Initialize the loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Move criterion to GPU
    criterion = criterion.to(rank)

    # Set the starting epoch for training
    starting_epoch = 0

    # Load checkpoint if specified
    if args.load_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1

    # Display model information. Only GPU 0 will display the information
    if rank == 0:
        # Display dataset information
        logger.info(f"Train Dataset size: {dataset_sizes['train']}.")
        logger.info(f"Scaling factor: {args.scaling_factor}.")
        # Display model configuration
        for key, val in model_config.items():
            print(f'{key}:\t{val}')

        logger.info(
            f"Initialized model with {get_model_size(model, False)} "
            f"total parameters, of which {get_model_size(model, True)} are learnable."
        )
        wandb.init(
            reinit=True,
            # set the wandb project where this run will be logged
            project="Scaling AI-Brain similarity 2",
            mode="offline", # GPUs are not connected to the internet. 
            # sync wandb with the cloud at the end of the job
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "Starting epoch": starting_epoch,
            "epochs": args.epochs,
            "Resume Training": args.load_checkpoint,
            "batch_size": args.batch_size,
            "Optimizer": args.optimizer, 
            "Total param": get_model_size(model, False),
            "Trainable param": get_model_size(model, True),
            "Train Dataset size": dataset_sizes['train'],
            "scaling factor": args.scaling_factor,
            }
        )

    return model, optimizer, criterion, dataset_loader, dataset_sizes, starting_epoch, wandb


def ddp_setup(rank: int, world_size: int) -> None:
    """
    Initialize the distributed environment for PyTorch DistributedDataParallel (DDP).

    Args:
        rank (int): Process rank within the distributed environment.
        world_size (int): Total number of processes participating in the distributed training.

    Returns:
        None
    """
    # Set the address and port for the master process
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group using NCCL as the backend
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Set the current CUDA device based on the process rank
    torch.cuda.set_device(rank)


def distributed_training(rank, world_size, command_line_args) -> None:
    """
    Performs distributed training using Distributed Data Parallel (DDP).

    Args:
        gpu_rank (int): The rank of the GPU on which the current process is running.
        world_size (int): The total number of GPUs available for distributed training.
        command_line_args (argparse.Namespace): Parsed command line arguments.

    Returns:
        None
    """
    ddp_setup(rank, world_size)
    if world_size > 1 and rank == 0:
        logger.info(f"Using DistributedDataParallel Training with {world_size} GPUs.")

    model, optimizer, criterion, dataset_loader, dataset_sizes, starting_epoch, wandb = load_train_objects(command_line_args, rank)
    model.to(rank)

    # Convert BatchNorm layers for synchronization in distributed training
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # Initialize DistributedDataParallel
    model = DDP(model, device_ids=[rank])

    # Initialize Trainer and perform training and testing
    if rank == 0:
        logger.info("Starting training...")
        trainer = Trainer(model, dataset_loader, dataset_sizes, optimizer, criterion, starting_epoch, command_line_args.epochs, command_line_args.save_every, wandb, rank)
    else:
        trainer = Trainer(model, dataset_loader, dataset_sizes, optimizer, criterion, starting_epoch, command_line_args.epochs, command_line_args.save_every, None, rank)
    trainer.train()
    trainer.test()

    # Clean up after training
    destroy_process_group()
    wandb.finish()


if __name__ == '__main__':
    config_parser = get_config_parser()
    command_line_args = config_parser.parse_args()

    if command_line_args.device == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available. Ensure your environment is GPU-enabled. "
            'Forcing device="cpu".'
        )
        command_line_args.device = "cpu"

    if command_line_args.device == "cpu":
        warnings.warn(
            "You are about to run on CPU, and may run out of memory shortly."
        )

    seed_experiment(command_line_args.seed)

    world_size = torch.cuda.device_count() # number of GPUs available
    mp.spawn(distributed_training, nprocs=world_size, args=(world_size, command_line_args))