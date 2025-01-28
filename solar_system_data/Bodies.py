import numpy as np
import os


class Body:
    """
    A class to represent a celestial body.
    Attributes:
    ----------
    name : str
        The name of the celestial body.
    mass : float, optional
        The mass of the celestial body in kg (default is 0).
    radius : float, optional
        The radius of the celestial body in km (default is 0).
    color : tuple, optional
        The color of the celestial body in RGB format (default is (255, 255, 255)).
    Methods:
    -------
    __init__(self, name: str, mass: float = 0, radius: float = 0, color=(255, 255, 255)):
        Initializes the Body with a name, mass, radius, and color.
    """

    def __init__(
        self,
        name: str,
        radius: float = 0,
        mass: float = 0,
        get_data=True,
        meters=False,
        color=(255, 255, 255),
    ):
        self.name = name

        if get_data:
            self.data = get_body_numpy(name)

            if meters:
                self.data *= 1000
        else:
            self.data = np.array([])

        self.mass = mass
        self.radius = radius
        self.color = color
        blender_object = None

        if self.name.lower() == "earth":
            self.color = (0, 0, 255)
        elif self.name.lower() == "mars":
            self.color = (255, 0, 0)
        elif self.name.lower() == "jupiter":
            self.color = (255, 128, 0)


def get_body_numpy(planet_id):
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
    mercury = get_body_numpy(199)
    venus = get_body_numpy(299)
    earth = get_body_numpy(399)
    mars = get_body_numpy(499)
    jupiter = get_body_numpy(599)
    saturn = get_body_numpy(699)
    uranus = get_body_numpy(799)
    neptune = get_body_numpy(899)

    return mercury, venus, earth, mars, jupiter, saturn, uranus, neptune
