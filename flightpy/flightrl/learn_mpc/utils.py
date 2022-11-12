import os
import numpy as np


class DataSaver:
    def __init__(self, folder) -> None:
        subpkg = "envtest"
        cwd = os.getcwd()
        self.dir = cwd[:cwd.find(subpkg)] + "flightmare/flightpy/datasets/" + folder + "/"

        self.metadata = self._read_metadata()

    def save_data(self, input_data, output_data, **kwargs):
        environment = kwargs["environment"]

        update_metadata = False
        for idx, data in enumerate([input_data, output_data]):
            file_to_save_path = f"{self.dir}{'input' if idx == 0 else 'output'}_{environment}.npy"
            if not os.path.exists(file_to_save_path):
                np.save(file_to_save_path, data)
                update_metadata = True
            elif kwargs["crashes"] < self.metadata[environment]:
                np.save(file_to_save_path, data)
                update_metadata = True
        if update_metadata:
            self._add_metadata(**kwargs)

    def _read_metadata(self):
        metadata = dict()
        with open(self.dir + "metadata.txt", "r+") as metadata_file:
            metadata_lines = metadata_file.read().splitlines()
            for line in metadata_lines:
                env = line.split()[2].rstrip(":")
                collisions = int(line.split()[-2])
                metadata[env] = collisions
        return metadata

    def _add_metadata(self, **kwargs):
        with open(self.dir + "metadata.txt", "a+") as metadata_file:
            metadata_file.write(f"\nDataset from {kwargs['environment']}: Stored {kwargs['size']} data sequences with overall {kwargs['crashes']} collisions")
        self.metadata[kwargs["environment"]] = kwargs["crashes"]
