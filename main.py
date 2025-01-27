import numpy as np
import init
from twob import two_body_analytical_update
from find_paths import iterate_permutations

def force(pos_a, pos_b, m_a, m_b):
    # G = 6.67408e-11
    G = 1.0

    # Force on point A due to point B
    F = G * m_a * m_b / np.linalg.norm(pos_b - pos_a)**3 * (pos_b - pos_a)
    return F


def pairwise_forces(pos, masses, G=1.0):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    distances_inv3 = np.where(distances > 0, distances**-3, 0)

    forces = G * diff * (distances_inv3[:, :, np.newaxis] * (masses[:, np.newaxis] * masses[np.newaxis, :])[:, :, np.newaxis])

    net_forces = np.sum(forces, axis=1) * -1
    return net_forces


def verlet_update(pos, vel, masses, dt, G=1.0):
    acc = pairwise_forces(pos, masses, G) / masses[:, np.newaxis]
    pos_next = pos + vel * dt + 0.5 * acc * dt**2
    acc_next = pairwise_forces(pos_next, masses, G) / masses[:, np.newaxis]
    vel_next = vel + 0.5 * (acc + acc_next) * dt

    return pos_next, vel_next


def sat_opening(pos_sat, pos_launch, normal_launch):
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
    - max_radius (float): Radius of the largest satellite-free cylinder in meters.
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

    # If there are no satellites, define a maximum radius (maybe based on atmospheric thickness)
    if len(distances) == 0:
        # Define maximum radius as 100 km
        max_radius = 100000
        return max_radius

    # Minimum distance is the largest possible radius without containing any satellite
    min_distance = np.min(distances)

    # Define a safety margin
    safety_margin = 100

    # Calculate max_radius by subtracting safety margin from min_distance
    max_radius = min_distance - safety_margin

    # Ensure the radius is non-negative
    max_radius = max(max_radius, 0)

    return max_radius


def get_launch_site(planets_pos, sats_pos, goes_idx):
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


if __name__ == "__main__":
    planets_pos, planets_vel, planets_mass = init.planets()
    sats_pos, sats_vel, goes_idx = init.satellites()
    n_sats = len(sats_pos)

    # Need to set this to whatever our start time is when initialising
    t0 = 0

    dt = 100
    t_max = t0 + 1e4

    sat_opening_thresh = 10

    # INIT VISUALISER HERE
    ...

    # VISUALISE INITIAL POSITIONS HERE
    ...
    for t in range(t0, t_max, dt):
        planets_pos, planets_vel = verlet_update(planets_pos, planets_vel,
                                                 planets_mass, dt, G=6.674e-11)

        # CALCULATE NEXT POS FOR SATELLITES HERE
        for sat in range(n_sats):
            _, _, sats_pos[sat], sats_vel[sat] = two_body_analytical_update(np.array([0.0, 0.0, 0.0]),
                                                                            np.array([0.0, 0.0, 0.0]),
                                                                            planets_mass[2],
                                                                            sats_pos[sat],
                                                                            sats_vel[sat],
                                                                            500,
                                                                            dt)

        # VISUALISE UPDATES POS' HERE
        ...

        pos_launch, point_above = get_launch_site(planets_pos, sats_pos, goes_idx)
        launch_normal = point_above - pos_launch
        # Checks for a candidate launch time.
        if sat_opening_thresh < sat_opening(sats_pos, pos_launch, launch_normal):

            # Save current permutation to a file
            path = "snapshots/" + str(t) + ".txt"
            np.savez(path, planets_pos=planets_pos,
                     planets_vel=planets_vel,
                     launch_normal=launch_normal + planets_pos[2])

    iterate_permutations()
