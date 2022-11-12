# NMPC Small Dataset #
The dataset was created by running Nonlinear MPC on **all** `hard` environments, and contains 14091 input-output sequences.

The whole dataset is compressed into one `.npz` file with the following structure:
- 100 2-dim. numpy arrays for the NMPC input
- 100 2-dim. numpy arrays for the NMPC output/controls

Each array can be referenced by its 'name': `(input|output)_hard_<environment_id>`. More information on each array can be found in [metadata file](metadata.txt).


## Input/Output Dimensionalities ##

By default, all data arrays have *first* dimensionality of 200 (since they are created with some pre-defined value before being filled up).

*Second* dimensions are:
- Each **input** arrays has *second* dimensionality of 111: 7-dim. state of 15 obstacles + 6-dim. drone state.
- Each **output** arrays has *second* dimensionality of 72: 3-dim. drone control (i.e., linear velocities) * prediction horizon (T-1) of 24.


## Important Notes ##

> **_NOTE 1:_** This is a 'small' dataset, meaning it stores only NMPC controls as output. For the full state output, please refer to 'full' dataset (tbd).

> **_NOTE 2:_** Each array is initially created with `np.empty()` (see [code](https://github.com/arseniid/agile_flight/blob/learn-mpc/envtest/ros/run_competition.py#L41-L44)) -> each array is filled up to the size of 200 with *random* values instead zeros.
