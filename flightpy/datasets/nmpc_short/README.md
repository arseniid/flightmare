# NMPC Short Dataset #
The dataset was created by running Nonlinear MPC on **all** `hard` environments, and contains 16945 input-output sequences.

The whole dataset is compressed into one `.npz` file with the following structure:
- 101 2-dim. numpy arrays for the NMPC input
- 101 2-dim. numpy arrays for the NMPC output

Each array can be referenced by its 'name': `(input|output)_hard_<environment_id>`. More information on each array can be found in [metadata file](metadata.txt).


## Input/Output Dimensionalities ##

The *first* dimensionality of all data arrays can be read from [metadata file](metadata.txt) (see *number of sequences* for each array).

*Second* dimensions are:
- Each **input** array has *second* dimensionality of 108: 7-dim. *relative* (i.e., to the drone state and in drone's body frame) state of 15 obstacles + 3-dim. direction to the goal (seen from the drone's body frame).
- Each **output** array has *second* dimensionality of 99: 9-dim. predicted NMPC output (i.e., drone's position + linear velocity + acceleration) * prediction horizon (T-1) of 11.

For example, `dataset_hard.npz['input_hard_18']` will be of dimension (157, 108).


## Important Notes ##

> **_NOTE 1:_** The dataset contains all computed NMPC outputs, meaning that the models, trained on it, can be used either as direct controllers (i.e., by taking only velocities as controls) as well as feed the predicted values as warm start for NMPC.

> **_NOTE 2:_** NMPC *Short* means that the horizon is only 12 steps long (for a time discretion of 0.08 seconds).
