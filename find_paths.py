import os
import numpy as np


def process_permutation(file_path):
    data = np.load(file_path)
    planets_pos = data['planets_pos']
    launch_normal = data['launch_normal']

    # PATHFINDING HERE
    ...


def iterate_permutations():
    dirpath = "snapshots"

    if not os.path.exists(dirpath):
        print(f"Directory '{dirpath}' does not exist.")
        return

    for permpath in os.listdir(dirpath):
        permutation = os.path.join(dirpath, permpath)
        process_permutation(permutation)


if __name__ == "__main__":
    iterate_permutations()
