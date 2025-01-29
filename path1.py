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

def test_path(file_path):
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

    # Check which planet the launch site is facing
    # Iterate over time steps

        # Iterate over all planets
    for i, planet_pos in enumerate(planets_pos):
        # Get the corresponding velocity for the current planet
        planet_vel = planets_vel[i]

        # Check if the launch site is facing the planet
        if launch_facing_planet(sat_pos, planet_pos, launch_normal):
            path1 = hohmann_transfer(sat_pos, planet_pos, mu_planets[8])
            print1 = print('Doing hohmann transfer to:', i)

            # Calculate the gravitational parameter (mu) for the planet
            mu_planet = calculate_mu(planets_mass[i], G)
            # Escape velocity calculation from Earth
            v_earth_escape = np.sqrt(2 * mu_planet / np.linalg.norm(planet_pos))

            # Semi-major axis of the transfer orbit
            a_orbit_transfer = (np.linalg.norm(planet_pos) + np.linalg.norm(planets_pos[i])) / 2
            v_orbital = np.sqrt(mu_planets[8] * (2 / np.linalg.norm(planet_pos) - 1 / a_orbit_transfer))

            # Calculate the initial velocity of the spacecraft
            initial_velocity = v_earth_escape + v_orbital

            # Gravitational assist (if not Jupiter, assuming index 3 is Jupiter)
            if i != 4:  # Skip Jupiter for gravitational assist
                v_planet = planet_vel
                print3 = print('v in:', initial_velocity)

                for r_closest in range(-100, 10):  # Calculate closest approach range
                    deflection_angle, v_out = gravitational_assist(mu_planet, initial_velocity, planet_vel, r_closest)
                    r_post_assist = planet_pos + r_closest * np.array([1, 0, 0])  # Update spacecraft position after assist
                    spacecraft_vel = v_out
                    print('deflection angle:', deflection_angle)

                # Update spacecraft velocity and position using gravitational influences
                    for j, second_planet_pos in enumerate(planets_pos):
                        distance_planet_spacecraft = r_post_assist - second_planet_pos
                        gravity_accel = -G * planets_mass[j] * distance_planet_spacecraft / (np.linalg.norm(distance_planet_spacecraft)**3)

                        # Update velocity and position iteratively
                        spacecraft_vel += gravity_accel * dt
                        sat_pos += spacecraft_vel * dt

                        #print('v out:', spacecraft_vel)
                    # Check if the trajectory leads towards Jupiter (you need to define `is_trajectory_to_jupiter`)
                    if is_trajectory_to_jupiter(r_post_assist, v_out, planets_pos[4], tolerance=1e9):
                        post_assist_path = hohmann_transfer(planet_pos, planets_pos[4], mu_sun)
                        print('post assist from:', planet_pos)
                        break
                    else:
                        print('there is no trajectory to jupiter from here')
                        closest_planet_idx = None
                        min_distance = float('inf')

                        for k, next_planet_pos in enumerate(planets_pos):
                             if k != 4:  # Skip Jupiter (index 4)
                                distance_to_jupiter = np.linalg.norm(planets_pos[4] - next_planet_pos)
                                if distance_to_jupiter < min_distance:
                                    min_distance = distance_to_jupiter
                                    closest_planet_idx = k
                    # Perform gravitational assist with the closest planet
                        if closest_planet_idx is not None:
                            closest_planet_pos = planets_pos[closest_planet_idx]
                            closest_planet_vel = planets_vel[closest_planet_idx]

                            print(f"Gravitational assist from planet {closest_planet_idx}, closest to Jupiter.")

                            # Perform assist and update trajectory
                            deflection_angle, v_new = gravitational_assist(mu_planets[closest_planet_idx], v_out, closest_planet_vel, r_closest=100)
                            r_post_assist = closest_planet_pos + 100 * np.array([1, 0, 0])  # Update spacecraft position after assist
                            v_out = v_new

                            if is_trajectory_to_jupiter(r_post_assist, v_out, planets_pos[4], tolerance=1e6):
                                print(f"Trajectory heading to Jupiter after assist from planet {closest_planet_idx}!")
                                break
                        else:
                            print("No planet found to assist. Trying Hohmann transfer to a planet closer to Jupiter.")
                            # Example: Perform a Hohmann transfer to Mars (index 2) to adjust the path
                            transfer_path = hohmann_transfer(r_post_assist, planets_pos[2], mu_sun)
                            print("Hohmann transfer initiated to adjust trajectory.")

                            # Update spacecraft position and velocity after Hohmann transfer
                            r_post_assist, v_out = transfer_path[0], transfer_path[1]


            break

    return 
