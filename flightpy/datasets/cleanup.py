import argparse
import os

import numpy as np


def clean_metadata(folder):
    clean_metadata = []
    for i in range(101):
        placeholder = f"Placeholder placeholder placeholder_{i}: placeholder 999 placeholder"
        clean_metadata.append(placeholder)

    metadata_path = f"{folder}/metadata.txt"
    with open(metadata_path, "r+") as metadata_file:
        metadata_lines = metadata_file.read().splitlines()

        for line in metadata_lines:
            line_list = line.split()
            env = line_list[2].rstrip(":")
            crashes = int(line_list[-2])
            idx = int(env.split("_")[-1])

            better_than = clean_metadata[idx]
            better_than_list = better_than.split()
            assert idx == int(better_than_list[2].rstrip(":").split("_")[-1])
            assert crashes < int(better_than_list[-2])

            clean_metadata[idx] = line

        assert len(clean_metadata) == 101
        sequences_overall = 0
        for clean_line in clean_metadata:
            sequences_overall += int(clean_line.split()[4])
        print(f"Dataset contains {sequences_overall} sequences.")

        metadata_file.seek(0)  # move the pointer to the beginning of the file
        metadata_file.write("\n".join(clean_metadata))
        metadata_file.truncate()  # removes the file contents after the specified number of bytes (if the size is not specified, the current position will be used)


def archive_dataset(folder):
    """
    Stores dataset as one compressed .npz file
    Note: Saves up to 50% of disk memory (compared to uncompressed `np.savez()`)
    """
    kwargs = dict()
    for i in range(101):
        for arr_key in [f"input_hard_{i}", f"output_hard_{i}"]:
            arr = np.load(f"{folder}/{arr_key}.npy")
            kwargs[arr_key] = arr

    dataset_path = f"{folder}/dataset_hard.npz"
    np.savez_compressed(dataset_path, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Small Python script for cleaning up the produced dataset")
    parser.add_argument(
        "--folder",
        help="Folder containing metadata.txt",
        required=False,
        default="nmpc"
    )
    parser.add_argument(
        "--no-metadata",
        help="Flag for NOT cleaning the metadata.txt file",
        required=False,
        action="store_true"
    )
    parser.add_argument(
        "--no-archive",
        help="Flag for NOT archiving the dataset .npy files into one .npz",
        required=False,
        action="store_true"
    )

    if os.getcwd().split("/")[-1] != "datasets":
        print(f"Please run the script from the 'agile_flight/flightmare/flightpy/datasets/' folder!")

    args = parser.parse_args()
    if not args.no_metadata:
        clean_metadata(folder=args.folder)
    if not args.no_archive:
        archive_dataset(folder=args.folder)
