import numpy as np
import os


class Planet:
    def __init__(self, name: str):
        self.name = name
        self.data = get_planet_numpy(name)

        self.color = (255, 255, 255)

        if self.name.lower() == "earth":
            self.color = (0, 0, 255)
        elif self.name.lower() == "mars":
            self.color = (255, 0, 0)
        elif self.name.lower() == "jupiter":
            self.color = (255, 128, 0)


def get_planet_numpy(planet_id):
    # Load the NumPy array from file

    planet_id = str(planet_id).lower()
    file_path = None

    # Get the directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    planet_data_dir = os.path.join(base_dir, "planet_data")

    if planet_id == "199" or planet_id == "mercury":
        file_path = os.path.join(planet_data_dir, "planet_199.npy")
    elif planet_id == "299" or planet_id == "venus":
        file_path = os.path.join(planet_data_dir, "planet_299.npy")
    elif planet_id == "399" or planet_id == "earth":
        file_path = os.path.join(planet_data_dir, "planet_399.npy")
    elif planet_id == "499" or planet_id == "mars":
        file_path = os.path.join(planet_data_dir, "planet_499.npy")
    elif planet_id == "599" or planet_id == "jupiter":
        file_path = os.path.join(planet_data_dir, "planet_599.npy")
    elif planet_id == "699" or planet_id == "saturn":
        file_path = os.path.join(planet_data_dir, "planet_699.npy")
    elif planet_id == "799" or planet_id == "uranus":
        file_path = os.path.join(planet_data_dir, "planet_799.npy")
    elif planet_id == "899" or planet_id == "neptune":
        file_path = os.path.join(planet_data_dir, "planet_899.npy")

    if file_path is None:
        raise ValueError(f"Invalid planet ID: {planet_id}")

    return np.load(file_path)


def get_all_numpy():
    # Load all planet NumPy arrays
    mercury = get_planet_numpy(199)
    venus = get_planet_numpy(299)
    earth = get_planet_numpy(399)
    mars = get_planet_numpy(499)
    jupiter = get_planet_numpy(599)
    saturn = get_planet_numpy(699)
    uranus = get_planet_numpy(799)
    neptune = get_planet_numpy(899)

    return mercury, venus, earth, mars, jupiter, saturn, uranus, neptune
