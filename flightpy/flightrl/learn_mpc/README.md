# Learning the (N)MPC module #
To train the learnable (N)MPC, you will need:
1. (N)MPC dataset (see available in ['datasets/'](../../datasets/) folder)
2. Compatible model (see available at [mpc_nn.py](mpc_nn.py) -- check the input/output dimensionalities!)
3. Various helping classes (note: these are quite general and usually do not require adaptations)
    - [NMPCDataset](mpc_dataset.py) represents a PyTorch wrapper around the .npz data
    - [Trainer](utils.py) class provides a general training routine in PyTorch
4. Training script [main.py](main.py)


## Training ##
To start training, run the training script as follows:

```python
python3 main.py [-h] [--hparam-search]
```

The script will train the model for a defined number of epochs, print intermediate losses and save the model inside the ['models/'](../../models/) folder.
Additionally, it will report Tensorboard logs to the ['runs/'](runs/) folder.

> **_NOTE:_** If the `--hparam-search` flag is given, no final training (as well as saving the model) will happen! Instead, the script will try to do simple
random search of the best hyperparameters (for 150 different sets of hyperparameters, for 5 epochs each).