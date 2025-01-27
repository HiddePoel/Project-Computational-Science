import os
import numpy as np
import math

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

# Calculate whether the rocket at time t is facing a planet
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

def calculate_mu(planet_mass, G):
    # Calculate mu constant for Hohmann assits
    return (planet_mass * G)/ 10**9


def process_permutation(file_path):
    data = np.load(file_path)
    planets_pos = data['planets_pos']
    planets_vel = data['planets_vel']
    planets_mass = data['planets_mass']
    launch_normal = data['launch_normal']
    sat_pos = data['sat_pos']

    t0 = 0

    dt = 0.1
    t_max = t0 + 100

    sat_opening_thresh = 10
    mu_sun = 1.327 * 10**11
    G = 6.674 * 10**-11

    for t in np.arange(t0, t_max, dt):

        for planet in planets_pos:
            # Check if the launch site is facing the planet
            if launch_facing_planet(sat_pos, planets_pos[planet], launch_normal):
                # Calculate the Hohmann transfer path
                planet1 = planets_pos['EARTH']
                planet2 = planets_pos[planet]

                path1 = hohmann_transfer(planet1, planet2, mu_sun)

                # Initialize spacecraft position and velocity
                spacecraft_pos = launch_normal

                # Calculate the mu of the assist planet
                mu_planets = {}
                mu_planets[planet] = calculate_mu(planet_mass[planet], G)

                # Calculate initial velocity needed for the spaceship to go to the first planet of the Hohmann transfer
                v_earth_escape = np.sqrt(2 * mu_planets['EARTH'] / planet1)
                a_orbit_transfer = (planet1, planets_pos[planet])/2
                v_orbital = np.sqrt(mu_sun * (2/planet1 - 1/a_orbit_transfer))

                initial_velocity = v_earth_escape + v_orbital


                # Check if the planet is not Jupiter before performing a gravitational assist
                if planet != 'JUPITER':
                    v_planet = planet_vel[planet]

                    for r_closest in range(1, 101):
                        # Measure the new angle of trajectory and velocity of the planet post-assist
                        deflection_angle, v_out = gravitational_assist(mu_planets[planet], initial_velocity, planets_vel[planet], r_closest)
                        r_post_assist = planets_pos[planet] + r_closest * np.array([1, 0, 0])
                        spacecraft_vel = v_out

                        # Compute graviational acceleration due to gravity assist
                        for _ in range(100):
                            gravity_accel = np.zeros(3)
                            for second_planet in planets_pos:
                                distance_planet_spacecraft = r_post_assist - planets_pos[second_planet]
                                gravity_accel -= G * planets_mass[other_planet] * distance_planet_spacecraft / (np.linalg.norm(distance_planet_spacecraft)**3)

                            # Update spacecraft velocity and position
                            spacecraft_vel += gravity_accel * dt
                            spacecraft_pos += spacecraft_vel * dt

                        if is_trajectory_to_jupiter(r_post_assist, v_out, planets_pos['JUPITER'], tolerance=1e6):
                            post_assist_path = hohmann_transfer(planet, planets_pos['JUPITER'], mu)



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
