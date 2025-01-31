import numpy as np
from twob import two_body_analytical_update
import os

import sys
sys.path.append(os.path.abspath('../tests'))
from optimized_twob import twobo


def opening(pos_sat, pos_launch, normal_launch, goes_idx):
    # calculates how large the opening is above launch location. We need to
    # somehow keep track of earth's orientation idk how. we can start with
    # the math to actually calculate the opening first

    # pos_launch = 3d pos
    # normal_launch = vector perpendicular to earth surface at pos_launch.

    # return the radius of the largest cilinder we can make in the atmosphere
    # in the direction of 'normal_launch' that doesn't contain a sattelite.
    """
    Calculates the radius of the largest cylinder aligned with the launch normal
    that does not contain any satellites.

    Parameters:
    - pos_sat (np.ndarray): Array of satellite positions, shape (N, 3).
    - pos_launch (np.ndarray): 3D position vector of the launch site, shape (3,).
    - normal_launch (np.ndarray): Vector perpendicular to Earth's surface at launch site, shape (3,).

    Returns:
    - max_radius (float): Radius of the largest satellite-free cylinder in kilometers meters.
    """
    # Validate inputs
    if not isinstance(pos_sat, np.ndarray) or pos_sat.ndim != 2 or pos_sat.shape[1] != 3:
        raise ValueError("pos_sat must be a NumPy array with shape (N, 3).")
    if not isinstance(pos_launch, np.ndarray) or pos_launch.shape != (3,):
        raise ValueError("pos_launch must be a NumPy array with shape (3,).")
    if not isinstance(normal_launch, np.ndarray) or normal_launch.shape != (3,):
        raise ValueError("normal_launch must be a NumPy array with shape (3,).")
    if np.linalg.norm(normal_launch) == 0:
        raise ValueError("normal_launch vector must be non-zero.")

    # Normalize the launch axis
    launch_axis = normal_launch / np.linalg.norm(normal_launch)

    # Vector from launch site to each satellite
    vec_to_sat = pos_sat - pos_launch  # Shape: (N, 3)

    # Compute cross product between vec_to_sat and launch_axis
    cross_prod = np.cross(vec_to_sat, launch_axis)  # Shape: (N, 3)

    # Compute perpendicular distances from satellites to the launch axis
    distances = np.linalg.norm(cross_prod, axis=1)  # Shape: (N,)
    distances[goes_idx] = 10000

    # Minimum distance is the largest possible radius without containing any satellite
    min_distance = np.min(distances)

    return min_distance


def launch_site(planets_pos, sats_pos, goes_idx):
    # For now we will set the launch site right below this geostationary
    # 'GOES 16' satellite
    goes_pos = sats_pos[goes_idx] * 1e3
    magnitude = np.linalg.norm(goes_pos)
    unit = goes_pos / magnitude
    earth_mean_radius = 6371000

    # Relative to center of earth
    launch_site = unit * earth_mean_radius
    point_above_site = unit * (earth_mean_radius + 1)
    return launch_site, point_above_site


def update(sats_pos, sats_vel, planets_mass, dt):
    new_pos = np.zeros_like(sats_pos)
    new_vel = np.zeros_like(sats_vel)
    for sat in range(len(sats_pos)):
        _, _, new_pos[sat], new_vel[sat] = two_body_analytical_update(np.array([0.0, 0.0, 0.0]),
                                                                      np.array([0.0, 0.0, 0.0]),
                                                                      planets_mass[2],
                                                                      sats_pos[sat, :],
                                                                      sats_vel[sat, :],
                                                                      500,
                                                                      dt)

    return new_pos, new_vel


def update_optimized(sats_pos, sats_vel, dt):
    for sat in range(len(sats_pos)):
        sats_pos[sat], sats_vel[sat] = twobo(sats_pos[sat], sats_vel[sat], 500, dt)
    return sats_pos, sats_vel
