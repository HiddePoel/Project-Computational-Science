file_path = '1.txt.npz'

data = np.load(file_path)
# List the keys in the .npz file
#print("Keys in the .npz file:", data.keys())

# Access individual arrays by key
for key in data.keys():
    print(f"{key}: {data[key]}")

planets_mass = np.array([0.33, 4.87, 5.97, 0.642, 1898.0, 568.0, 86.8, 102.0, 988416.0])

def convert_to_radius(planet_pos, planet1_idx):
    # Return radius (in meters)
    r1 = np.linalg.norm(planets_pos[planet1_idx])
    return r1 / 1000

def calculate_mu(planet_mass, G):
    # Calculate mu constant for Hohmann assits (in m/s)
    return (planet_mass * G)/ 10**3

def hohmann_transfer(r1, r2, mu):
    """Calculate Hohmann transfer orbit from one planet to the next.
    Inputs:
        r1 = radius of planet 1 (distance from the Sun)
        r2 = radius of planet 2
        mu = graviational parameter of the sun
    """
    # Calculate oribtal param of transfer elipse
    # Semi-major axis (a_transfer)
    a_transfer = (r1 + r2) / 2

    # Eccentricity (e)
    e_transfer = (r2 - r1) / (r2 + r1)

    # Calculate velocities
    # Earth and Jupiter orbit velocity
    v1 = np.sqrt(mu/r1)
    v2 = np.sqrt(mu/r2)

    # Velocity and first and second burn (first and second orbit)
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

    # Outgoing velocity in the planet's frame (same magnitude, rotated by deflection angle)
    rotation_matrix = np.array([
        [math.cos(deflection_angle), -math.sin(deflection_angle), 0],
        [math.sin(deflection_angle), math.cos(deflection_angle), 0],
        [0, 0, 1]
    ])

    v_out_planet = np.dot(rotation_matrix, v_in_planet)

    # Convert back to Sun's reference frame
    v_out_sun = v_out_planet + v_planet

    return deflection_angle, v_out_sun

def launch_facing_planet(launch_pos, planet_pos, normal_launch):
    """Check whether the launch site is facing any planets.
    Inputs:
        launch_pos = position (vector) of launch site
        planet_pos = position of the planet it could be facing
        normal_launch =
    """

    # Calculate the vector from the launch site to Jupiter
    vector_to_jupiter = planet_pos - launch_pos

    # Normalize both the normal_launch vector and vector_to_jupiter
    normal_launch_unit = normal_launch / np.linalg.norm(normal_launch)
    vector_to_jupiter_unit = vector_to_jupiter / np.linalg.norm(vector_to_jupiter)

    # Calculate the dot product between the normalized vectors
    dot_product = np.dot(normal_launch_unit, vector_to_jupiter_unit)

    # If the dot product is close to 1, the vectors are nearly aligned (facing each other)
    if dot_product > 0.9:
        return True
    return False

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

def process_permutation(file_path):
    data = np.load(file_path)
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
        mass_planet = mass * 10**24
        mu_planet = calculate_mu(mass_planet, G)
        mu_planets.append(mu_planet)

    # Calcualte all the radii of the planets
    r_planets = []
    for planet_idx, _ in enumerate(planets_pos):  # Use planet_idx to get the index of each planet
        r_planet = convert_to_radius(planets_pos, planet1_idx=planet_idx)
        r_planets.append(r_planet)

    # Chekc which planet is closest between Earth and Jupiter
    closest_planet = find_closest_planet_between(planets_pos[2], planets_pos[4], planets_pos)

    # Check whether the angle between the planet and Jupiter is less than 45 degrees
    if is_facing_jupiter(planets_pos[2], planets_pos[4], planets_pos[closest_planet]):
        # If planet is in the path to jupiter, perform hohmann transfer
        first_hohmann = hohmann_transfer(r_planets[2], r_planets[closest_planet], mu_planets[8])
        print('Doing hohmann transfer to planet', closest_planet)

        # Calculate escape velocity to assist planet
        v_earth_escape = np.sqrt(2 * mu_planets[closest_planet] / np.linalg.norm(planets_pos))
        # Semi-major axis of the transfer orbit
        a_orbit_transfer = (np.linalg.norm(planets_pos) + np.linalg.norm(planets_pos[closest_planet])) / 2
        v_orbital = np.sqrt(mu_planets[8] * (2 / np.linalg.norm(planets_pos) - 1 / a_orbit_transfer))

        # Calculate the initial velocity of the spacecraft and the planet
        initial_velocity = v_earth_escape + v_orbital
        v_planet = planets_vel[closest_planet]

        # Perform gravitational assist to Jupiter from the closet planet
        # Initialize r_closest and step size
        r_closest = 0  # Initial closest approach distance
        step = 1  # Step size for adjusting r_closest
        max_attempts = 10  # Prevent infinite loop
        attempts = 0

        while attempts < max_attempts:
            deflection_angle, v_out = gravitational_assist(mu_planet, initial_velocity, planets_vel, r_closest)
            r_post_assist = planets_pos[closest_planet] + r_closest * np.array([1, 0, 0])  # Update spacecraft position
            spacecraft_vel = v_out[closest_planet]
            print(f"Attempt {attempts + 1}: r_closest = {r_closest}, Deflection Angle = {deflection_angle}")

            if is_trajectory_to_jupiter(r_post_assist, spacecraft_vel, r_planets[4], tolerance=1e6):
                print(f"Optimal r_closest found: {r_closest}")
                print(f"Trajectory heading to Jupiter after assist from planet {closest_planet_idx}!")
                break  # Stop loop if we found a good trajectory

            # Adjust r_closest based on results
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
        permutation = os.path.join(dirpath, permpath)
        process_permutation(permutation)


if __name__ == "__main__":
    iterate_permutations()
