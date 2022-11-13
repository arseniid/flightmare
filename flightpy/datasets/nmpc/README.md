# NMPC Small Dataset #
The dataset was created by running Nonlinear MPC on **all** `hard` environments, and contains 14091 input-output sequences.

The whole dataset is compressed into one `.npz` file with the following structure:
- 100 2-dim. numpy arrays for the NMPC input
- 100 2-dim. numpy arrays for the NMPC output/controls

Each array can be referenced by its 'name': `(input|output)_hard_<environment_id>`. More information on each array can be found in [metadata file](metadata.txt).


## Input/Output Dimensionalities ##

The *first* dimensionality of all data arrays can be read from [metadata file](metadata.txt).

*Second* dimensions are:
- Each **input** arrays has *second* dimensionality of 111: 7-dim. state of 15 obstacles + 6-dim. drone state.
- Each **output** arrays has *second* dimensionality of 72: 3-dim. drone control (i.e., linear velocities) * prediction horizon (T-1) of 24.

For example, `dataset_hard.npz['input_hard_18']` will be of dimension (134, 111).

## Important Notes ##

> **_NOTE 1:_** This is a 'small' dataset, meaning it stores only NMPC controls as output. For the full state output, please refer to 'full' dataset (tbd).
