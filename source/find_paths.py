import numpy as np
import pandas as pd
import math
import os

def find_closest_planet_between(earth_pos, jupiter_pos, planets_pos):
    """Find the closest planet to Earth between Earth and Jupiter."""
    # Calculate vector from Earth to Jupiter
    vector_ej = jupiter_pos - earth_pos

    # Find planets between Earth and Jupiter based on orbital distance
    planets_between = {}
    for i, pos in enumerate(planets_pos):
        # Skip Earth (index 2) and Jupiter (index 4)
        if i in [2, 4]:
            continue
        distance_from_sun = np.linalg.norm(pos)
        earth_distance = np.linalg.norm(earth_pos)
        jupiter_distance = np.linalg.norm(jupiter_pos)

        if earth_distance < distance_from_sun < jupiter_distance:
            planets_between[i] = distance_from_sun

    if not planets_between:
        return None

    # Find the planet closest to Earth
    closest_planet_idx = min(planets_between, key=planets_between.get)

    return closest_planet_idx

def convert_to_radius(planets_pos, planet1_idx):
    """Return radius (in meters)"""
    r1 = np.linalg.norm(planets_pos[planet1_idx])
    return r1 / 1000

def calculate_mu(planet_mass, G):
    """ Calculate mu constant for Hohmann assits (in m/s)"""
    return (planet_mass * G)/ 10**3

def hohmann_transfer(r1, r2, mu):
    """Calculate Hohmann transfer orbit from one planet to the next.
    Inputs:
        r1: radius of planet 1 (distance from the Sun)
        r2: radius of planet 2
        mu: graviational parameter of the sun
    """
    # Calculate oribtal param of transfer elipse
    # Semi-major axis (a_transfer)
    a_transfer = (r1 + r2) / 2

    # Eccentricity
    e_transfer = (r2 - r1) / (r2 + r1)

    # Calculate velocities
    v1 = np.sqrt(mu/r1)
    v2 = np.sqrt(mu/r2)

    # Velocity and first and second burn (first and second planet orbit)
    v_periapsis = np.sqrt(mu * ((2/r1) - (1/a_transfer)))
    delta_v1 = v_periapsis - v1

    v_apoapsis = np.sqrt(mu * ((2/r2) - (1/a_transfer)))
    delta_v2 = v_apoapsis - v2

    total_delta_v = delta_v1 + delta_v2

    return total_delta_v

def gravitational_assist(mu_planet, v_in_spacecraft, v_planet, r_closest):
    """Calculate graviational assist of planet x
    Inputs:
        mu_planet = graviational parameter of assist planet
        r_closest = closest approach (radius) of spaceship
        r_spaceship = radius of planet around the sun
        v_planet = velocity of planet around the sun
    """
    # Calculate velocity of spaceship relative to the planet
    v_in_planet = v_in_spacecraft - v_planet
    v_in_magnitude = np.linalg.norm(v_in_planet)

    deflection_angle = 2 * math.asin(1 / (1 + (r_closest * v_in_magnitude**2) / mu_planet))

    # Outgoing velocity in the planet's frame
    rotation_matrix = np.array([
        [math.cos(deflection_angle), -math.sin(deflection_angle), 0],
        [math.sin(deflection_angle), math.cos(deflection_angle), 0],
        [0, 0, 1]
    ])

    v_out_planet = np.dot(rotation_matrix, v_in_planet)

    # Convert back to Sun's reference frame
    v_out_sun = v_out_planet + v_planet

    return deflection_angle, v_out_sun


def is_trajectory_to_jupiter(r_post_assist, v_out, r_jupiter, tolerance):
    """
    Check if the spacecraft's trajectory intersects with Jupiter's fixed position.
    Inputs:
        r_post_assist: Position of the spacecraft after the assist
        v_out: Velocity vector of the spacecraft after the assist
        r_jupiter: Fixed position of Jupiter
        tolerance: Distance threshold for "intersecting"
    """
    # Solve for t where trajectory gets close to Jupiter
    t = np.dot(r_jupiter - r_post_assist, v_out) / np.dot(v_out, v_out)

    # Find the closest point on the trajectory to Jupiter
    closest_point = r_post_assist + t * v_out
    distance_to_jupiter = np.linalg.norm(closest_point - r_jupiter)

    return distance_to_jupiter < tolerance

def is_facing_jupiter(earth_pos, jupiter_pos, closest_planet_pos):
    """Check if the angle between Earth-Jupiter vector and Earth-closest planet vector is <= 45 degrees."""
    # Calculate vectors between Earth and Jupiter, and Earth and the closest planet
    vector_ej = jupiter_pos - earth_pos
    vector_ec = closest_planet_pos - earth_pos
    # Calculate the angle between the two vectors
    angle = calculate_angle(vector_ej, vector_ec)
    return angle <= 45.0

import numpy as np


def calculate_angle(vector1, vector2):
    """
    Calculates the angle (in degrees) between two 3D vectors using the dot product formula.

    Parameters:
    - vector1 (numpy.ndarray): A 3D vector represented as a NumPy array.
    - vector2 (numpy.ndarray): Another 3D vector represented as a NumPy array.

    Returns:
    - float: The angle between the two vectors in degrees.
    """

    # Normalize both vectors to unit length
    unit_v1 = vector1 / np.linalg.norm(vector1)  # Unit vector of vector1
    unit_v2 = vector2 / np.linalg.norm(vector2)  # Unit vector of vector2

    # Compute the dot product between the two unit vectors
    dot_product = np.dot(unit_v1, unit_v2)

    # Clamp the dot product value to the valid range [-1, 1] to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute the angle in radians using the inverse cosine (arccos) function
    angle_rad = np.arccos(dot_product)

    # Convert the angle from radians to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def process_permutation(file_path):
    data = np.load(file_path, allow_pickle=True)
    planets_pos = data['planets_pos']
    planets_vel = data['planets_vel']
    launch_normal = data['launch_normal']

    planets_mass = np.array([0.33, 4.87, 5.97, 0.642, 1898.0, 568.0, 86.8, 102.0, 988416.0])

    # Initialize everything
    t0 = 0
    dt = 0.1
    t_max = t0 + 100
    sat_pos = launch_normal
    G = 6.674 * 10**-11

    # Calculate all the mu of the planets
    mu_planets = []
    for mass in planets_mass:
        # Mass of planets in kg
        mass_planet = mass * 10**22
        mu_planet = calculate_mu(mass_planet, G)
        mu_planets.append(mu_planet)

    # Calcualte all the radii of the planets
    r_planets = []
    for planet_idx, _ in enumerate(planets_pos):
        r_planet = convert_to_radius(planets_pos, planet1_idx=planet_idx)
        r_planets.append(r_planet)

    # Chekc which planet is closest between Earth and Jupiter
    closest_planet = find_closest_planet_between(planets_pos[2], planets_pos[4], planets_pos)

    # Check whether the angle between the planet and Jupiter is less than 45 degrees
    if is_facing_jupiter(planets_pos[2], planets_pos[4], planets_pos[closest_planet]):
        # If planet is in the path to Jupiter, perform Hohmann transfer
        first_hohmann = hohmann_transfer(r_planets[2], r_planets[closest_planet], mu_planets[8])
        print('Doing hohmann transfer to planet', closest_planet)

        # Calculate escape velocity to assist planet from Earth
        v_earth_escape = np.sqrt(2 * mu_planets[closest_planet] / np.linalg.norm(planets_pos))
        a_orbit_transfer = (np.linalg.norm(planets_pos) + np.linalg.norm(planets_pos[closest_planet])) / 2
        v_orbital = np.sqrt(mu_planets[8] * (2 / np.linalg.norm(planets_pos) - 1 / a_orbit_transfer))

        # Calculate the initial velocity of the spacecraft and the planet
        initial_velocity = v_earth_escape + v_orbital
        v_planet = planets_vel[closest_planet]

        # Perform gravitational assist to Jupiter from the closet planet
        r_closest = 10e2
        step = 10e4
        max_attempts = 1000
        attempts = 0

        # Loop over different r_closest, as it dictates the outgoing deflection angle
        while attempts < max_attempts:
            deflection_angle, v_out = gravitational_assist(mu_planet, initial_velocity, planets_vel[closest_planet], r_closest)
            r_post_assist = planets_pos[closest_planet] + r_closest * np.array([1, 0, 0])
            spacecraft_vel = v_out

            # Check if the deflection angle sends the spacecraft to Jupiter, otherwise try a different approach distance
            if is_trajectory_to_jupiter(r_post_assist, spacecraft_vel, r_planets[4], tolerance=1e6):
                print(f"Trajectory heading to Jupiter after assist from planet {closest_planet_idx}!")
                break

            r_closest += step
            attempts += 1

        else:
            print("No suitable r_closest found within the max attempts. Consider adjusting parameters.")

    else:
        print('Cannot use planet', closest_planet, 'for gravitational assist')

    return

def iterate_permutations():
    dirpath = "snapshots"

    if not os.path.exists(dirpath):
        print(f"Directory '{dirpath}' does not exist.")
        return

    for permpath in os.listdir(dirpath):
        if permpath == ".DS_Store":
            continue

        permutation = os.path.join(dirpath, permpath)
        process_permutation(permutation)

if __name__ == "__main__":
    iterate_permutations()
