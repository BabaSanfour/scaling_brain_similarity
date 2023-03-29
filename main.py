import json
import time
import wandb
import warnings
import torch
import torch.nn as nn
from resnet import resnet
from data.data_loader import dataloader
from config import get_config_parser
from utils import seed_experiment, get_model_size

def train_network(model, criterion, optimizer, epochs, dataset_loader, dataset_sizes, device, wandb, valid=True):
    best_acc = -1.0 
    list_trainLoss, list_trainAcc, list_valLoss, list_valAcc  = [], [], [], []
    since = time.time()
    steps=0
    best_acc_steps=0
    for epoch in range(epochs):
        print(f"====== Epoch {epoch} ======>")
        n_correct = 0 #correct predictions train
        running_loss = 0.0 # train loss
        loss = 0.0
        with torch.set_grad_enabled(True):
            model.train()
            for inputs_, labels_ in dataset_loader["train"]:
                steps+=1
                inputs = inputs_.to(device)
                labels = labels_.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                n_correct += (torch.max(outputs, 1)[1].view(labels.data.size()) == labels.data).sum().item()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            train_acc = 100. * n_correct/dataset_sizes["train"]
            train_loss = running_loss / dataset_sizes["train"]
            print(f"== [TRAIN] Epoch: {epoch}, Accuracy: {train_acc:.3f} ==>, Loss: {train_loss:.3f} ==>,")
            list_trainLoss.append(train_loss)
            list_trainAcc.append(train_acc)

        if valid:
            with torch.no_grad():
                model.eval()
                n_dev_correct=0
                running_loss = 0.0
                loss = 0.0
                for inputs_, labels_ in dataset_loader["valid"]:
                    inputs = inputs_.to(device)
                    labels = labels_.to(device)
                    outputs = model(inputs)
                    n_dev_correct += (torch.max(outputs, 1)[1].view(labels.data.size()) == labels.data).sum().item()
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                valid_acc = 100. * n_dev_correct/dataset_sizes["valid"]
                valid_loss = running_loss / dataset_sizes["valid"]
                list_valLoss.append(valid_loss)
                list_valAcc.append(valid_acc)
                print(f"== [VALID] Epoch: {epoch}, Accuracy: {valid_acc:.3f} ==>, Loss: {valid_loss:.3f} ==>,")
                if valid_acc> best_acc:
                    best_acc= valid_acc
                    best_model = model
                    best_acc_steps = steps
            wandb.log({"Val Loss": valid_loss, "Val Acc": valid_acc, 
                    "Train Loss": train_loss, "Train Acc": train_acc, "steps":steps})
        else:
            wandb.log({"Train Loss": train_loss, "Train Acc": train_acc, "steps":steps})
    print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 3600} h {(time_elapsed % 3600) // 60} m {time_elapsed % 60} s')
    print(f'BEST VALID ACC: {valid_acc:.3f}')
    print(f'Total steps: {steps}.')
    print(f'Steps required to reach best performance: {best_acc_steps}.')
    return best_model, wandb
    
def test_network(model, dataset_loader, dataset_sizes, device):
    model.eval()
    n_dev_correct=0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for inputs_, labels_ in dataset_loader['test']:
        inputs = inputs_.to(device)
        labels = labels_.to(device)
        outputs = model(inputs)
        n_dev_correct += (torch.max(outputs, 1)[1].view(labels.data.size()) == labels.data).sum().item()
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
    avg_test_acc = 100. * n_dev_correct/dataset_sizes["test"]
    avg_test_loss = test_loss / dataset_sizes["test"]
    print(f'TEST LOSS: {avg_test_loss:.3f}')
    print(f'TEST Accuracy (Top1): {avg_test_acc:.3f}')
    return avg_test_acc

def save_network_weights(model_ft,  file) :
    """Save the network after training"""
    state = model_ft.state_dict()
    torch.save(state, file)

if __name__ == '__main__':
    parser = get_config_parser()
    args = parser.parse_args()

    # Check for the device
    if (args.device == "cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available, make that your environment is "
            "running on GPU"
            'Forcing device="cpu".'
        )
        args.device = "cpu"

    if args.device == "cpu":
        warnings.warn(
            "You are about to run on CPU, and might run out of memory shortly"
        )

    # Seed the experiment, for repeatability
    seed_experiment(args.seed)

    start = time.time()
    # Load model
    if args.model_config is not None:
        print(f'Loading model config from {args.model_config}')
        with open(args.model_config) as f:
            model_config = json.load(f)
    else:
        raise ValueError('Please provide a model config json')
    for key, val in model_config.items():
        print(f'{key}:\t{val}')
    print('############################################')
    model = resnet(**model_config)
    model.to(args.device)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    best_acc=-1.0
    wandb.init(
            # set the wandb project where this run will be logged
            project="Scaling AI-Brain similarity",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "Optimizer": args.optimizer, 
            "Total param": get_model_size(model, False),
            "Trainable param": get_model_size(model, True),
            }
        )


    # Optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    # model size
    print(
        f"Initialized model with {get_model_size(model, False)} "
        f"total parameters, of which {get_model_size(model, True)} are learnable."
    )
    #DATA
    dataset_loader, dataset_sizes = dataloader(args.batch_size, args.scaling_factor)

    print(
        f"Loaded data of around {int(1300/args.scaling_factor)} picture"
        f"scaling factor {args.scaling_factor}."
    )

    ###Training & Validation###
    model_ft, wandb = train_network(model, criterion, optimizer, args.epochs, dataset_loader, dataset_sizes, args.device, wandb)
    ###Testing###
    acc=test_network(model_ft, dataset_loader, dataset_sizes, args.device)
    wandb.log ({"Test Acc": acc})


    #  Save weights after training
    save_network_weights(model, args.model_name)
