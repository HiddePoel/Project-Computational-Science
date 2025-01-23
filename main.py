import numpy as np
import pandas as pd

# Open and read the data files
planet_path = 'planet_data.txt'
satellite_path = 'tle_active.txt'

with open(satellite_path, 'r') as satellite_file:
    lines = satellite_file.readlines()

# Read test tle file and organize into name, line1 and line2
sat_table = []

for i in range(0, len(lines), 3):
    name = lines[i].strip()
    line_1 = lines[i+1].strip()
    line_2 = lines[i+2].strip()

    sat_table.append([name, line_1, line_2])

columns = ['Name', 'Line 1', 'Line 2']
satellite_df = pd.DataFrame(sat_table, columns=columns)

# Read planet data into df
plan_table = []
current_data = {}
with open(planet_path, 'r') as planet_file:
    for line in planet_file:
        line = line.strip()
        # An empty line means the end of data of one planet
        if not line:
            if current_data:
                plan_table.append(current_data)
                current_data = {}
        else:
            key, value = line.split(':',1)
            current_data[key.strip()] = value.strip()
    if current_data:
        plan_table.append(current_data)

planet_df = pd.DataFrame(plan_table)


def init_solar(solar_file):
    initial_coord = {}
    results = []

    for _, row in solar_file.iterrows():
        # Separate info in RA and DEC (hour, minutes, seconds)
        ra_parts = row['RA'].split()
        dec_parts = row['DEC'].split()

        # Convert RA and DEC to floats and then degrees
        ra_h, ra_min, ra_sec = map(float, ra_parts)
        dec_h, dec_min, dec_sec = map(float, dec_parts)
        distance = float(row['Distance'])  

        ra_deg = ra_h * 15 + ra_min * 15 / 60 + ra_sec * 15 / 3600
        dec_deg = (abs(dec_h) + dec_min / 60 + dec_sec / 3600) * (-1 if dec_h < 0 else 1)

        # Convert degrees to radians
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)

        # Calculate Cartesian coordinates
        x = distance * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance * np.cos(dec_rad) * np.sin(ra_rad)
        z = distance * np.sin(dec_rad)

        # Store initial coordinates
        initial_coord[row['Planet']] = {'x': x, 'y': y, 'z': z}

    # Adjust coordinates to be relative to Earth
    earth_position = initial_coord.get('EARTH', {'x': 0, 'y': 0, 'z': 0})
    for planet, coords in initial_coord.items():
        if planet != 'EARTH':
            initial_coord[planet]['x'] -= earth_position['x']
            initial_coord[planet]['y'] -= earth_position['y']
            initial_coord[planet]['z'] -= earth_position['z']

    # Calculate orbital velocity and inclination adjustments
    for _, row in solar_file.iterrows():
        planet = row['Planet']
        distance = float(row['Distance'])
        inclination = np.radians(float(row['Inclination']))
        orb_vel = float(row['Orbital Velocity'])

        # Get the initial coordinates
        coords = initial_coord.get(planet, {'x': 0, 'y': 0, 'z': 0})
        x, y, z = coords['x'], coords['y'], coords['z']

        # Apply rotation matrix for inclination
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(inclination), -np.sin(inclination)],
            [0, np.sin(inclination), np.cos(inclination)]
        ])
        rotated_coords = np.dot(rotation_matrix, np.array([x, y, z]))

        # Calculate velocity
        theta = np.arctan2(y, x)
        v_x = -orb_vel * np.sin(theta)
        v_y = orb_vel * np.cos(theta)
        v_z = 0
        velocity = np.array([v_x, v_y, v_z])
        rotated_velocity = np.dot(rotation_matrix, velocity)

        results.append({
            'Planet': planet,
            'Position X (kme6)': rotated_coords[0],
            'Position Y (kme6)': rotated_coords[1],
            'Position Z (kme6)': rotated_coords[2],
            'Velocity X (km/s)': rotated_velocity[0],
            'Velocity Y (km/s)': rotated_velocity[1],
            'Velocity Z (km/s)': rotated_velocity[2]
        })

    planet_df_processed = pd.DataFrame(results)
    return planet_df_processed


def init_satellites(sat_file, year, month, day, hour, minute, second):
    # Calculate xyz, Vx Vy Vz for all the satellites

    results = []

    # Get all the positions and velocities of the satellites
    for index, row in sat_file.iterrows():
        name = row['Name']
        line1 = row['Line 1']
        line2 = row['Line 2']

        satellite = Satrec.twoline2rv(line1, line2)

        # Loop over every day of one month (can change to hours, and can do it over much longer time)
        # Convert the date into julian date
        for day in range(0,1):
            jd, fr = jday(year, month, day, hour, minute, second)

            # Measure the satellite position (r) and velocity (r) at the given date
            e, r, v = satellite.sgp4(jd, fr)

            if e == 0:
                results.append({
                    "Name": name,
                    "Time": f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}",
                    "Position (km)": list(r),
                    "Velocity (km/s)": list(v)
                })
            else:
                print(f"Error propagating {name}: Error code {e}")

    columns = ['Name', 'Time', 'Position(km)', 'Velocity(km/s)']
    rv_df = pd.DataFrame(results)

    rv_df[['Position X (km)', 'Position Y (km)', 'Position Z (km)']] = pd.DataFrame(rv_df['Position (km)'].tolist(), index=rv_df.index)
    rv_df[['Velocity X (km/s)', 'Velocity Y (km/s)', 'Velocity Z (km/s)']] = pd.DataFrame(rv_df['Velocity (km/s)'].tolist(), index=rv_df.index)
    rv_df = rv_df.drop(columns=['Position (km)', 'Velocity (km/s)'])
    return rv_df



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


# def accel(pos_a, pos_b, m_a, m_b):
#     return force(pos_a, pos_b, m_a, m_b) / m_a


# def accel_all(idx, pos, mass):
#     mask = np.ones_like(mass, dtype=bool)
#     mask[idx] = False

#     pos_B = pos[mask]
#     m_B = mass[mask]

#     pos_a = pos[idx]
#     m_a = mass[idx]

#     a = 0.0
#     for i in range(len(pos_B)):
#         a += accel(pos_a, pos_B[i], m_a, m_B[i])
#     return a


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

# used for earth sim
# is this really needed?
def twobody_next_pos():
    ...

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
