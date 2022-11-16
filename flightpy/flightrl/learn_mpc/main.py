import argparse
import os
import random

from torch.nn import MSELoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from mpc_dataset import NMPCDataset
from mpc_nn import MPCFullLearned, MPCControlLearned, MPCControlLearnedBN, MPCControllLearnedSmall,MPCControlLearnedSmallBN
from utils import Trainer


def get_dataloaders(dataset_folder="nmpc", batch_size=8):
    dataset = NMPCDataset(root_dir=dataset_folder)
    train_set, val_set = random_split(dataset, [round(0.8 * len(dataset)), round(0.2 * len(dataset))])  # 80% / 20% split

    # num_workers should be 1, otherwise there are various zipFile errors due to concurrent read of the same .npz archive
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=1
    )
    return train_loader, val_loader


def random_hyperparam_search(epochs=5):
    models = [MPCControlLearned, MPCControlLearnedBN, MPCControllLearnedSmall, MPCControlLearnedSmallBN]
    batch_sizes = [4, 8, 16, 32, 64, 128]
    lr_rates = [1e-5, 1e-2]
    lr_decays = [0, 1e-1]
    momenta = [0, 0.9]
    optimizers = [Adam, SGD]
    # TODO: non-linearities? different loss functions?

    best_val_loss = 1e5
    for i in range(30):
        print(f"\n=== Random search #{i} starts ===")
        model = random.choice(models)
        if model in [MPCControlLearnedBN, MPCControlLearnedSmallBN]:
            # BatchNorm is recommended to use with batch_size >= 32
            batch_sizes = [16, 32, 64, 128]

        hyperparams = {
            "batch_size": random.choice(batch_sizes),
            "learning_rate": random.uniform(*lr_rates),
            "lr_decay": random.uniform(*lr_decays),
            "momentum": random.uniform(*momenta),
            "optimizer": random.choice(optimizers),
            "loss_fn": MSELoss,
            "logger": SummaryWriter(log_dir=f"runs/hparam_search/exp{i}"),
        }
        train_loader, val_loader = get_dataloaders(batch_size=hyperparams["batch_size"])

        trainer = Trainer(model, train_dataloader=train_loader, val_dataloader=val_loader, **hyperparams)
        val_losses = trainer.train(epochs=epochs)

        val_losses_better = [x <= best_val_loss for x in val_losses]
        val_loss_okay = [x <= 1.0 for x in val_losses]
        if any(val_losses_better + val_loss_okay):
            print(f"Current best val loss: {best_val_loss}")
            print(f"Current val losses: {val_losses}")
            best_val_loss = min(val_losses)
            print("Model and hyperparameters are:")
            print(f"    - {model}")
            print("\n".join([f"    - {k}: {v}"for k, v in hyperparams.items()]))


def train_model(epochs):
    model = MPCControlLearnedBN

    hyperparams = {
        "batch_size": 64,
        "learning_rate": 7e-3,
        "lr_decay": 3e-3,
        "momentum": 0.0,
        "optimizer": Adam,
        "loss_fn": MSELoss,
        "logger": SummaryWriter(),
    }
    train_loader, val_loader = get_dataloaders(batch_size=hyperparams["batch_size"])

    trainer = Trainer(
        model, train_dataloader=train_loader, val_dataloader=val_loader, **hyperparams,
    )
    trainer.train(epochs=epochs)

    trainer.save_model(directory="../../models", file_name="nmpc_control_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NMPC model")
    parser.add_argument(
        "--hparam-search",
        help="Flag for hyperparameter search",
        required=False,
        action="store_true"
    )
    args = parser.parse_args()

    if os.getcwd().split("/")[-1] != "learn_mpc":
        raise FileNotFoundError("Saving of the trained model depends on the current working directory. "
                                "Please run the script from the 'agile_flight/flightmare/flightpy/flightrl/learn_mpc/' folder!")

    if args.hparam_search:
        print(random_hyperparam_search())
    else:
        train_model(epochs=20)
