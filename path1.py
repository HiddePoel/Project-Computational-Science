import numpy as np
import pandas as pd
import math

# Function to calculate Hohmann transfer orbit

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
    a_transfer = (r_closest + r_planet) / 2

    # Outgoing velocity in the planet's frame (same magnitude, rotated by deflection angle)
    v_out_planet = np.array([
        v_in_planet[0] * math.cos(deflection_angle) - v_in_planet[1] * math.sin(deflection_angle),
        v_in_planet[0] * math.sin(deflection_angle) + v_in_planet[1] * math.cos(deflection_angle),
        0
    ])

    # Convert back to Sun's reference frame
    v_out_sun = v_out_planet + v_planet

    return deflection_angle, v_out_s

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


# given all the input:
planets_pos, planets_vel, planets_mass = init.planets()
sats_pos, sats_vel, goes_idx = init.satellites()

# Need to set this to whatever our start time is when initialising
t0 = 0

dt = 0.1
t_max = t0 + 100

sat_opening_thresh = 10

mu_sun = 1.87e3 #(sun graviational param idk what it is)
mu_earth = 3.98e4 # earth graviational param -- CHECK IT
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

            # Initialize spacecraft position and velocity
            spacecraft_pos = planet1

            # Calculate initial velocity needed for the spaceship to go to the first planet of the Hohmann transfer
            v_earth_escape = np.sqrt(2 * mu_earth / r_earth)
            a_orbit_transfer = (r_earth, planet_pos[planet]['Distance'])/2
            v_orbital = np.sqrt(mu_sun * (2/r_earth - 1/a_orbit_transfer))

            initial_velocity = v_earth_escape + v_orbital

            # Check if the planet is not Jupiter before performing a gravitational assist
            if planet != 'JUPITER':
                v_planet = planet_pos[planet]['Velocity']
                for r_closest in range(1, 101):
                    # Measure the new angle of trajectory and velocity of the planet post-assist
                    deflection_angle, v_out = gravitational_assist(mu_assist, initial_velocity, v_planet, r_closest)
                    r_post_assist = r_planet + r_closest * np.array([1, 0, 0])
                    spacecraft_vel = v_out

                    # Compute graviational acceleration due to gravity assist
                    for _ in range(100):
                        accel_gravity = np.zeros(3)
                        for second_planet in planets_pos:
                            distance_planet_spacecraft = r_post_assist - planet_pos[second_planet]
                            gravity_accel -= G * planets_mass[other_planet] * distance_planet_spacecraft / (np.linalg.norm(distance_planet_spacecraft)**3)

                        # Update spacecraft velocity and position
                        spacecraft_vel += gravity_accel * dt
                        spacecraft_pos += spacecraft_vel * dt

                    if is_trajectory_to_jupiter(r_post_assist, v_out, planet_pos['JUPITER'], tolerance=1e6):
                        post_assist_path = hohmann_transfer(planet, planet_pos['JUPITER'], mu)
