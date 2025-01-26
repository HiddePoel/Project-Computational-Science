import numpy as np
import pandas as pd
import math

# Function to calculate Hohmann transfer orbit

def hohmann_transfer(r1, r2, mu):
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

def graviational_assit(mu_planet, v_in, r_closest):
    """Calculate graviational assist of planet x
    Inputs:
        mu_planet = graviational parameter of assist planet
        v_in = velocity of spaceship when entering the assist planet
        r_closest = closest approach (radius)
    """
    deflection_angle = 2 * math.asin(1 / (1 + (r_closest * v_in**2) / mu_planet))
    v_out = v_in

    return deflection_angle, v_out

# Calculate whether the rocket at time t is facing a planet
def launch_facing_planet(launch_pos, planet_pos, normal_launch):
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


# given all the input:
planets_pos, planets_vel, planets_mass = init.planets()
sats_pos, sats_vel, goes_idx = init.satellites()

# Need to set this to whatever our start time is when initialising
t0 = 0

dt = 0.1
t_max = t0 + 100

sat_opening_thresh = 10

mu = 1.87e3 #(sun graviational param idk what it is)

    # INIT VISUALISER HERE
...

    # VISUALISE INITIAL POSITIONS HERE
    ...
for t in range(t0, t_max, dt):
    planets_pos, planets_vel = verlet_update(planets_pos, planets_vel, planets_mass, dt, G=6.674e-11)

    # CALCULATE NEXT POS FOR SATELLITES HERE
    sats_pos = ...

        # VISUALISE UPDATES POS' HERE
    ...

        # Checks for a candidate launch time.
    if sat_opening_thresh < sat_opening():
            # GET THE LAUNCH NORMAL VECTOR HERE
        launch_normal = ...

    for planet in planets_pos:
        # Check if the launch site is facing the planet
        if launch_facing_planet(goes_pos, planets_pos[planet], launch_normal):
            # Calculate the Hohmann transfer path
            planet1 = planets_pos['EARTH']
            planet2 = planets_pos[planet]

            path1 = hohmann_transfer(planet1, planet2, mu)

            # Check if the planet is not Jupiter before performing a gravitational assist
            if planet != 'JUPITER':
                for r_closest in range(1, 101):  
                    deflection_angle, v_out = gravitational_assist(mu_assist, v_in, r_closest)
                    r_post_assist = r_planet + r_closest * np.array([1, 0, 0])
                    if is_trajectory_to_jupiter(r_post_assist, v_out, planet_pos['JUPITER'], tolerance=1e6):
                        post_assist_path = hohmann_transfer(planet, planet_pos['JUPITER'], mu)
