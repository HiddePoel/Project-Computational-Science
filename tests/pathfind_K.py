import os
import numpy as np
import math
import matplotlib.pyplot as plt

def hohmann_transfer(r1, r2, mu):
    """
    Calculate the total delta-v required for a Hohmann transfer orbit from planet 1 to planet 2.
    
    Parameters:
        r1 (float): Radius of planet 1's orbit around the Sun in km.
        r2 (float): Radius of planet 2's orbit around the Sun in km.
        mu (float): Gravitational parameter of the Sun in km^3/s^2.
        
    Returns:
        tuple: (delta_v1, delta_v2, total_delta_v)
            delta_v1 (float): Velocity change for the first burn in km/s.
            delta_v2 (float): Velocity change for the second burn in km/s.
            total_delta_v (float): Total velocity change in km/s.
    """
    # Semi-major axis of transfer orbit
    a_transfer = (r1 + r2) / 2.0

    # Orbital velocities of the planets
    v1 = np.sqrt(mu / r1)
    v2 = np.sqrt(mu / r2)

    # Velocities at periapsis and apoapsis of transfer orbit
    v_periapsis = np.sqrt(mu * (2.0 / r1 - 1.0 / a_transfer))
    v_apoapsis = np.sqrt(mu * (2.0 / r2 - 1.0 / a_transfer))

    # Delta-v calculations
    delta_v1 = v_periapsis - v1
    delta_v2 = v2 - v_apoapsis
    total_delta_v = delta_v1 + delta_v2

    return delta_v1, delta_v2, total_delta_v

def gravitational_assist(mu_planet, v_in_spacecraft, v_planet, r_closest):
    """
    Calculate the gravitational assist (slingshot) of a planet.
    
    Parameters:
        mu_planet (float): Gravitational parameter of the assist planet in km^3/s^2.
        v_in_spacecraft (np.ndarray): Incoming velocity vector of the spacecraft in km/s.
        v_planet (np.ndarray): Velocity vector of the assist planet in km/s.
        r_closest (float): Closest approach distance to the assist planet in km.
        
    Returns:
        tuple: (deflection_angle_degrees, v_out_spacecraft)
            deflection_angle_degrees (float): Deflection angle in degrees.
            v_out_spacecraft (np.ndarray): Outgoing velocity vector of the spacecraft in km/s.
    """
    # Velocity of spacecraft relative to the planet
    v_inf = v_in_spacecraft - v_planet
    v_inf_mag = np.linalg.norm(v_inf)
    
    # Calculate deflection angle using the hyperbolic trajectory formula
    deflection_angle = 2 * math.asin(1 / (1 + (r_closest * v_inf_mag**2) / mu_planet))
    deflection_angle_degrees = np.degrees(deflection_angle)
    
    # Assuming the deflection occurs in the plane of incoming velocity
    # Calculate the perpendicular unit vector to v_inf
    if v_inf_mag == 0:
        raise ValueError("Incoming velocity magnitude is zero; cannot perform assist.")
    
    # Normalize incoming velocity
    v_inf_unit = v_inf / v_inf_mag
    
    # Define a perpendicular vector (assuming 2D for simplicity)
    perp_unit = np.array([-v_inf_unit[1], v_inf_unit[0], 0])
    
    # Rotate the incoming velocity by the deflection angle to get outgoing velocity
    rotation_matrix = np.array([
        [math.cos(deflection_angle), -math.sin(deflection_angle), 0],
        [math.sin(deflection_angle),  math.cos(deflection_angle), 0],
        [0, 0, 1]
    ])
    
    v_out_relative = rotation_matrix @ v_inf
    v_out_spacecraft = v_out_relative + v_planet  # Convert back to Sun's reference frame
    
    return deflection_angle_degrees, v_out_spacecraft

def is_facing_Jupiter(earth_pos, jupiter_pos, closest_planet_pos):
    """
    Check if the angle between Earth-Jupiter vector and Earth-closest planet vector is <= 45 degrees.
    
    Parameters:
        earth_pos (np.ndarray): Position vector of Earth in km.
        jupiter_pos (np.ndarray): Position vector of Jupiter in km.
        closest_planet_pos (np.ndarray): Position vector of the closest planet in km.
        
    Returns:
        bool: True if angle <= 45 degrees, else False.
    """
    vector_ej = jupiter_pos - earth_pos
    vector_ec = closest_planet_pos - earth_pos
    angle = calculate_angle(vector_ej, vector_ec)
    return angle <= 45.0

def calculate_angle(vector1, vector2):
    """
    Calculate the angle in degrees between two vectors.
    
    Parameters:
        vector1 (np.ndarray): First vector.
        vector2 (np.ndarray): Second vector.
        
    Returns:
        float: Angle in degrees.
    """
    unit_v1 = vector1 / np.linalg.norm(vector1)
    unit_v2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_v1, unit_v2)
    # Clamp the dot product to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def find_closest_planet_between(earth_pos, jupiter_pos, planets_pos):
    """
    Find the closest planet to Earth between Earth and Jupiter.
    
    Parameters:
        earth_pos (np.ndarray): Position vector of Earth in km.
        jupiter_pos (np.ndarray): Position vector of Jupiter in km.
        planets_pos (dict): Dictionary of planet positions with planet names as keys.
        
    Returns:
        str or None: Name of the closest planet or None if no such planet exists.
    """
    # Calculate vector from Earth to Jupiter
    vector_ej = jupiter_pos - earth_pos
    distance_ej = np.linalg.norm(vector_ej)
    
    # Identify planets between Earth and Jupiter based on orbital distance
    planets_between = {}
    for planet, pos in planets_pos.items():
        if planet.upper() in ['EARTH', 'JUPITER']:
            continue  # Skip Earth and Jupiter
        distance = np.linalg.norm(pos)
        earth_distance = np.linalg.norm(earth_pos)
        jupiter_distance = np.linalg.norm(jupiter_pos)
        if earth_distance < distance < jupiter_distance:
            planets_between[planet] = distance
    
    if not planets_between:
        return None  # No planets between Earth and Jupiter
    
    # Find the planet with the minimum distance to Earth
    closest_planet = None
    min_distance = np.inf
    for planet, distance in planets_between.items():
        planet_vector = planets_pos[planet] - earth_pos
        distance_to_earth = np.linalg.norm(planet_vector)
        if distance_to_earth < min_distance:
            min_distance = distance_to_earth
            closest_planet = planet
    
    return closest_planet

def is_trajectory_to_jupiter(r_post_assist, v_out, r_jupiter, tolerance=1e6):
    """
    Check if the spacecraft's trajectory intersects with Jupiter's position within a tolerance.
    
    Parameters:
        r_post_assist (np.ndarray): Position of the spacecraft after assist in km.
        v_out (np.ndarray): Velocity vector of the spacecraft after assist in km/s.
        r_jupiter (np.ndarray): Fixed position of Jupiter in km.
        tolerance (float): Distance threshold in km for intersection.
        
    Returns:
        bool: True if trajectory intersects within tolerance, else False.
    """
    # Vector from spacecraft post-assist to Jupiter
    delta_r = r_jupiter - r_post_assist
    
    # Project delta_r onto velocity vector
    t = np.dot(delta_r, v_out) / np.dot(v_out, v_out)
    
    if t < 0:
        return False  # Jupiter is behind the spacecraft
    
    # Closest point on trajectory to Jupiter
    closest_point = r_post_assist + t * v_out
    distance_to_jupiter = np.linalg.norm(closest_point - r_jupiter)
    
    return distance_to_jupiter <= tolerance

def calculate_mu(planet_mass, G):
    """
    Calculate the gravitational parameter (mu) for a planet.
    
    Parameters:
        planet_mass (float): Mass of the planet in kg.
        G (float): Gravitational constant in km^3/kg/s^2.
        
    Returns:
        float: Gravitational parameter in km^3/s^2.
    """
    return planet_mass * G

def process_snapshot(planets_pos, planets_vel, planets_mass, launch_normal, sat_pos, mu_sun, G):
    """
    Process a single snapshot to find the best path from Earth to Jupiter.
    
    Parameters:
        planets_pos (dict): Dictionary of planet positions with planet names as keys in km.
        planets_vel (dict): Dictionary of planet velocities with planet names as keys in km/s.
        planets_mass (dict): Dictionary of planet masses with planet names as keys in kg.
        launch_normal (np.ndarray): Normal vector of the launch site.
        sat_pos (np.ndarray): Position vector of the satellite (Earth) in km.
        mu_sun (float): Gravitational parameter of the Sun in km^3/s^2.
        G (float): Gravitational constant in km^3/kg/s^2.
        
    Returns:
        dict: Details of the best path found.
    """
    earth_pos = planets_pos['EARTH']
    jupiter_pos = planets_pos['JUPITER']
    
    # Find the closest planet between Earth and Jupiter
    closest_planet_name = find_closest_planet_between(earth_pos, jupiter_pos, planets_pos)
    
    if closest_planet_name is None:
        # No planet between Earth and Jupiter; perform direct transfer
        delta_v1, delta_v2, total_delta_v = hohmann_transfer(
            r1=np.linalg.norm(earth_pos),
            r2=np.linalg.norm(jupiter_pos),
            mu=mu_sun
        )
        path = {
            'type': 'Direct Transfer',
            'delta_v1': delta_v1,
            'delta_v2': delta_v2,
            'total_delta_v': total_delta_v,
            'assist_planets': [],
            'path_points': [earth_pos, jupiter_pos]
        }
        return path
    
    # Get the position of the closest planet
    closest_planet_pos = planets_pos[closest_planet_name]
    
    # Check the angle condition
    if is_facing_Jupiter(earth_pos, jupiter_pos, closest_planet_pos):
        # Angle condition met; perform gravitational assist via closest planet
        # Calculate Hohmann transfer to closest planet
        r1 = np.linalg.norm(earth_pos)
        r2 = np.linalg.norm(closest_planet_pos)
        delta_v1, delta_v2, total_delta_v1 = hohmann_transfer(r1, r2, mu_sun)
        
        # Calculate mu for the assist planet
        mu_assist = calculate_mu(planets_mass[closest_planet_name], G)
        
        # Assume spacecraft is launched into Earth orbit; apply delta_v1
        # For simplicity, assume launch_normal is aligned with Earth's velocity
        # Spacecraft's initial velocity vector
        v_earth = np.linalg.norm(planets_vel['EARTH'])
        v_spacecraft_initial = planets_vel['EARTH'] + (delta_v1 * (launch_normal / np.linalg.norm(launch_normal)))
        
        # Perform gravitational assist at closest planet
        # Calculate closest approach distance (r_closest)
        # Use the planet's radius plus a buffer (e.g., 100 km)
        planet_radii = {
            'VENUS': 6051.8,   # km
            'MARS': 3389.5,    # km
            'SATURN': 58232.0   # km
            # Add other planets if needed
        }
        r_closest = planet_radii.get(closest_planet_name.upper(), 100.0) + 100.0  # km
        
        try:
            deflection_angle_deg, v_out_spacecraft = gravitational_assist(
                mu_planet=mu_assist,
                v_in_spacecraft=v_spacecraft_initial,
                v_planet=planets_vel[closest_planet_name],
                r_closest=r_closest
            )
        except ValueError as e:
            print(f"Gravitational assist error: {e}")
            # Fallback to direct transfer
            delta_v1, delta_v2, total_delta_v = hohmann_transfer(
                r1=np.linalg.norm(earth_pos),
                r2=np.linalg.norm(jupiter_pos),
                mu=mu_sun
            )
            path = {
                'type': 'Direct Transfer',
                'delta_v1': delta_v1,
                'delta_v2': delta_v2,
                'total_delta_v': delta_v1 + delta_v2,
                'assist_planets': [],
                'path_points': [earth_pos, jupiter_pos]
            }
            return path
        
        # Update spacecraft's velocity after assist
        spacecraft_vel_after_assist = v_out_spacecraft
        
        # Check if the trajectory now points towards Jupiter
        # For simplicity, assume a direct Hohmann transfer from assist planet to Jupiter
        r3 = np.linalg.norm(planets_pos[closest_planet_name])
        r4 = np.linalg.norm(jupiter_pos)
        delta_v3, delta_v4, total_delta_v2 = hohmann_transfer(r3, r4, mu_sun)
        
        # Total delta-v is the sum of all burns
        total_delta_v_total = total_delta_v1 + total_delta_v2
        
        # Record the path points
        path_points = [earth_pos, closest_planet_pos, jupiter_pos]
        
        path = {
            'type': 'Gravitational Assist',
            'delta_v1': delta_v1,
            'delta_v2': delta_v2,
            'delta_v3': delta_v3,
            'delta_v4': delta_v4,
            'total_delta_v': total_delta_v_total,
            'assist_planets': [closest_planet_name],
            'path_points': path_points
        }
        return path
    else:
        # Angle condition not met; perform direct transfer to Jupiter
        delta_v1, delta_v2, total_delta_v = hohmann_transfer(
            r1=np.linalg.norm(planets_pos['EARTH']),
            r2=np.linalg.norm(planets_pos['JUPITER']),
            mu=mu_sun
        )
        path = {
            'type': 'Direct Transfer',
            'delta_v1': delta_v1,
            'delta_v2': delta_v2,
            'total_delta_v': delta_v1 + delta_v2,
            'assist_planets': [],
            'path_points': [planets_pos['EARTH'], planets_pos['JUPITER']]
        }
        return path

def process_permutation(file_path):
    """
    Process a single snapshot file to find the best path from Earth to Jupiter.
    
    Parameters:
        file_path (str): Path to the snapshot file (.npz).
        
    Returns:
        dict: Details of the path found.
    """
    # Load data from the snapshot file
    try:
        data = np.load(file_path, allow_pickle=True)
        planets_pos = data['planets_pos'].item()      # Dictionary {planet: position vector in km}
        planets_vel = data['planets_vel'].item()      # Dictionary {planet: velocity vector in km/s}
        planets_mass = data['planets_mass'].item()    # Dictionary {planet: mass in kg}
        launch_normal = data['launch_normal']          # Vector (assumed to be a unit vector)
        sat_pos = data['sat_pos']                      # Vector (assumed to be Earth's position)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    # Initialize parameters
    mu_sun = 1.32712440018e11  # km^3/s^2
    G = 6.67430e-20             # km^3/kg/s^2

    # Find the best path for this snapshot
    path = process_snapshot(
        planets_pos=planets_pos,
        planets_vel=planets_vel,
        planets_mass=planets_mass,
        launch_normal=launch_normal,
        sat_pos=sat_pos,
        mu_sun=mu_sun,
        G=G
    )
    
    return path

def iterate_snapshots():
    """
    Iterate over all snapshot files in the 'snapshots' directory and find the best paths.
    """
    dirpath = "snapshots"

    if not os.path.exists(dirpath):
        print(f"Directory '{dirpath}' does not exist.")
        return

    all_paths = []

    for permpath in os.listdir(dirpath):
        permutation = os.path.join(dirpath, permpath)
        if not permutation.endswith('.npz'):
            continue  # Skip non-'.npz' files
        path = process_permutation(permutation)
        if path is None:
            continue  # Skip if processing failed
        path['snapshot'] = permpath
        all_paths.append(path)
        print(f"Processed snapshot: {permpath}")
        print(f"Path Type: {path['type']}")
        print(f"Total Delta-V: {path['total_delta_v']:.3f} km/s")
        print(f"Assist Planets: {path['assist_planets']}")
        print("-" * 40)
    
    if not all_paths:
        print("No valid snapshots processed.")
        return
    
    # Find the path with the minimum total_delta_v
    best_path = min(all_paths, key=lambda x: x['total_delta_v'])
    
    print("Best Path Found:")
    print(f"Snapshot: {best_path['snapshot']}")
    print(f"Path Type: {best_path['type']}")
    print(f"Total Delta-V: {best_path['total_delta_v']:.3f} km/s")
    print(f"Assist Planets: {best_path['assist_planets']}")
    
    # Optional: Plot the path
    plot_path(best_path)
    
    return best_path

def plot_path(path):
    """
    Plot the trajectory from Earth to Jupiter, including gravitational assists if any.
    
    Parameters:
        path (dict): Details of the path.
    """
    if not path or 'path_points' not in path:
        print("No path points to plot.")
        return
    
    path_points = path['path_points']
    path_points = np.array(path_points)
    
    plt.figure(figsize=(8, 8))
    plt.plot(0, 0, 'yo', label='Sun')  # Sun at origin
    
    # Plot Earth
    earth_pos = path_points[0]
    plt.plot(earth_pos[0], earth_pos[1], 'bo', label='Earth')
    
    # Plot Assist Planets
    for planet in path['assist_planets']:
        planet_pos = path_points[1]
        plt.plot(planet_pos[0], planet_pos[1], 'go', label=f'{planet}')
    
    # Plot Jupiter
    jupiter_pos = path_points[-1]
    plt.plot(jupiter_pos[0], jupiter_pos[1], 'ro', label='Jupiter')
    
    # Draw trajectory
    plt.plot(path_points[:,0], path_points[:,1], 'k--', label='Trajectory')
    
    plt.xlabel('X Position (km)')
    plt.ylabel('Y Position (km)')
    plt.title('Spacecraft Trajectory from Earth to Jupiter')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main():
    """
    Main function to execute the path finding process.
    """
    best_path = iterate_snapshots()
    
    if best_path:
        print("\nBest Mission Details:")
        print(f"Snapshot: {best_path['snapshot']}")
        print(f"Path Type: {best_path['type']}")
        print(f"Total Delta-V: {best_path['total_delta_v']:.3f} km/s")
        print(f"Assist Planets: {best_path['assist_planets']}")
    else:
        print("No valid mission paths found.")

if __name__ == "__main__":
    main()
