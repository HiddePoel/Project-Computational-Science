import numpy as np
import init


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


# used for earth sim
def twobody_update(pos1, pos2, vel1, vel2, mass1, mass2, dt):
    """
    Compute the next positions of a two-body system using the Verlet integration method.

    Parameters:
    - pos1, pos2: Current positions of the two bodies (numpy arrays).
    - vel1, vel2: Current velocities of the two bodies (numpy arrays).
    - mass1, mass2: Masses of the two bodies.
    - dt: Time step.

    Returns:
    - pos1_next, pos2_next: Updated positions.
    - vel1_next, vel2_next: Updated velocities.
    """
    # Compute gravitational force on body 1 due to body 2
    F = force(pos1, pos2, mass1, mass2)
    
    # Compute accelerations (Newton's third law)
    a1 = F / mass1
    a2 = -F / mass2  
    
    # Update positions
    pos1_next = pos1 + vel1 * dt + 0.5 * a1 * dt**2
    pos2_next = pos2 + vel2 * dt + 0.5 * a2 * dt**2
    
    # Compute forces at next positions
    F_next = force(pos1_next, pos2_next, mass1, mass2)
    
    # Compute new accelerations
    a1_next = F_next / mass1
    a2_next = -F_next / mass2
    
    # Update velocities
    vel1_next = vel1 + 0.5 * (a1 + a1_next) * dt
    vel2_next = vel2 + 0.5 * (a2 + a2_next) * dt
    
    return pos1_next, pos2_next, vel1_next, vel2_next


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


# def main_solar():
#     pos, vel, m = init.planets()

#     # time to sim
#     tts = 100
#     dt = 0.1

#     pos_next, vel_next = verlet_update(pos, vel, m, dt)
#     for step in range(tts - 1 // dt):
#         # vis_solar(pos)
#         pos, vel = verlet_update(pos_next, vel_next, m, dt)


# def main_earth():
#     pos, vel = init.satellites()

#     # time to sim
#     tts = 100
#     dt = 0.1

#     opening_thresh = 10
#     launcht_candidates = []

#     for step in range(tts // dt):
#         vis_earth(pos)
#         twobody_update()
#         if opening_thresh <= sat_opening():
#             # snapshot current time and position of satellites and earth
#             # add this to launcht_candidates or somewhere else.
#             ...


def get_launch_site(planets_pos, sats_pos, goes_idx):
    # For now we will set the launch site right below this geostationary
    # 'GOES 16' satellite
    goes_pos = sats_pos[goes_idx]
    magnitude = np.linalg.norm(goes_pos)
    unit = goes_pos / magnitude
    earth_mean_radius = 6371000

    # Relative to center of earth
    launch_site = unit * earth_mean_radius
    point_above_site = unit * (earth_mean_radius + 1)
    return planets_pos[2] + launch_site, planets_pos[2] + point_above_site


if __name__ == "__main__":
    planets_pos, planets_vel, planets_mass = init.planets()
    sats_pos, sats_vel, goes_idx = init.satellites()

    # Need to set this to whatever our start time is when initialising
    t0 = 0

    dt = 0.1
    t_max = t0 + 100

    sat_opening_thresh = 10

    # INIT VISUALISER HERE
    ...

    # VISUALISE INITIAL POSITIONS HERE
    ...
    for t in range(t0, t_max, dt):
        planets_pos, planets_vel = verlet_update(planets_pos, planets_vel,
                                                 planets_mass, dt, G=6.674e-11)

        # CALCULATE NEXT POS FOR SATELLITES HERE
        sats_pos = ...

        # VISUALISE UPDATES POS' HERE
        ...

        # Checks for a candidate launch time.
        if sat_opening_thresh < sat_opening():
            # GET THE LAUNCH NORMAL VECTOR HERE
            launch_normal = ...

            # Save current permutation to a file
            path = "snapshots/" + str(t) + ".txt"
            np.savez(path, positions=planets_pos, launch_normal=launch_normal)

            # EITHER SPAWN A SUBPROCESS AND FIND A PATH NOW OR PROCESS ALL OF
            # snapshots DIR AFTER.
            ...

    ...
