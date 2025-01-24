import numpy as np


def init_solar():
    # Initialize position(3d), velocity(3d), and mass of celestial bodies not
    # including satellites. So sun, planets, moons.

    ...

    # return pos, vel, mass


def init_satellites():
    # same as above.
    ...


def force(pos_a, pos_b, m_a, m_b):
    # G = 6.67408e-11
    G = 1

    # Force on point A due to point B
    F = G * m_a * m_b / np.linalg.norm(pos_b - pos_a)**3 * (pos_b - pos_a)
    return F


def force_all(idx, pos, mass):
    # calculate the force on body with index 'idx', due to all
    # the other bodies (with indexes =/= 'idx'). 'pos' and 'mass' are arrays.
    mask = np.ones_like(mass, dtype=bool)
    mask[idx] = False

    pos_B = pos[mask]
    m_B = mass[mask]

    pos_a = pos[idx]
    m_a = mass[idx]

    F = np.array((0.0, 0.0, 0.0))
    for i in range(len(pos_B)):
        F += force(pos_a, pos_B[i], m_a, m_B[i])

    return F


def accel(pos_a, pos_b, m_a, m_b):
    return force(pos_a, pos_b, m_a, m_b) / m_a


def accel_all(idx, pos, mass):
    mask = np.ones_like(mass, dtype=bool)
    mask[idx] = False

    pos_B = pos[mask]
    m_B = mass[mask]

    pos_a = pos[idx]
    m_a = mass[idx]

    a = 0.0
    for i in range(len(pos_B)):
        a += accel(pos_a, pos_B[i], m_a, m_B[i])
    return a


def verlet_update(poss, vels, masses, dt):
    N = len(poss)

    accs = np.zeros_like(vels, dtype=np.float64)
    for idx in range(N):
        accs[idx] = accel_all(idx, poss, masses)

    pos_next = np.zeros_like(poss, dtype=np.float64)
    for idx in range(N):
        pos_next[idx] = poss[idx] + vels[idx] * dt + 0.5 * accs[idx] * dt**2

    acc_next = np.zeros_like(accs)
    for idx in range(N):
        acc_next[idx] = accel_all(idx, pos_next, masses)

    vel_next = np.zeros_like(vels, dtype=np.float64)
    for idx in range(N):
        vel_next[idx] = vels[idx] + (accs[idx] + acc_next[idx]) / 2 * dt

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
 
 
def vis_earth(current_pos):
    ...


def vis_solar(current_pos):
    ...


def main_solar():
    pos, vel, m = init_solar()

    # time to sim
    tts = 100
    dt = 0.1

    # acc = np.zeros_like(pos, dtype=np.float64)
    # for idx in range(len(pos)):
    #     acc[idx] = accel_all(idx, pos, m)

    pos_next, vel_next = verlet_update(pos, vel, m, dt)
    for step in range(tts - 1 // dt):
        # vis_solar(pos)
        pos, vel = verlet_update(pos_next, vel_next, m, dt)


def main_earth():
    pos, vel, m = init_satellites()

    # time to sim
    tts = 100
    dt = 0.1

    opening_thresh = 10
    launcht_candidates = []

    for step in range(tts // dt):
        vis_earth(pos)
        twobody_update()
        if opening_thresh <= sat_opening():
            # snapshot current time and position of satellites and earth
            # add this to launcht_candidates or somewhere else.
            ...


if __name__ == "__main__":
    ...
