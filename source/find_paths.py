import os
import numpy as np


SNAPSHOTS_DIR = "../snapshots"


def process_permutation(file_path):
    data = np.load(file_path)
    planets_pos = data['planets_pos']
    planets_vel = data['planets_vel']
    launch_normal = data['launch_normal']

    # PATHFINDING HERE (TODO)
    ...


def iterate_permutations():
    if not os.path.exists(SNAPSHOTS_DIR):
        print(f"Directory '{SNAPSHOTS_DIR}' does not exist.")
        return

    for snapname in os.listdir(SNAPSHOTS_DIR):
        snapshot = os.path.join(SNAPSHOTS_DIR, snapname)
        process_permutation(snapshot)


if __name__ == "__main__":
    iterate_permutations()
