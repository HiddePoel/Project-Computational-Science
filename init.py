import numpy as np
from sgp4.api import Satrec, SatrecArray, jday
import requests


TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
INIT_YEAR = 2025
INIT_MONTH = 1
INIT_DAY = 1
INIT_HOUR = 0
INIT_MINUTE = 0
INIT_SECOND = 0


def satellites():
    # Fetch TLE data and transform to SatrecArray
    try:
        response = requests.get(TLE_URL)
        response.raise_for_status()
        tle_data = response.text.splitlines()

        satellites = []
        for i in range(0, len(tle_data), 3):
            if i + 2 < len(tle_data):
                name = tle_data[i].strip()
                line1 = tle_data[i + 1].strip()
                line2 = tle_data[i + 2].strip()
                try:
                    satrec = Satrec.twoline2rv(line1, line2)
                    satellites.append(satrec)
                except Exception as e:
                    print(f"Error parsing TLE for satellite '{name}': {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching TLE data: {e}")

    # Calculate initial position and velocities of every satellite
    jd, fr = jday(INIT_YEAR, INIT_MONTH, INIT_DAY, INIT_HOUR, INIT_MINUTE, INIT_SECOND)
    jd = np.array([jd])
    fr = np.array([fr])
    satarr = SatrecArray(satellites)
    e, pos, vel = satarr.sgp4(jd, fr)

    print(pos, vel)
    return pos, vel


def planets():
    planet_data = {
        "MERCURY": {"distance": 57.96, "mass": 0.330, "inclination": 7.0, "orbital velocity": 47.4},
        "VENUS": {"distance": 108.26, "mass": 4.87, "inclination": 3.4, "orbital velocity": 35.0},
        "EARTH": {"distance": 149.6, "mass": 5.97, "inclination": 0, "orbital velocity": 29.8},
        "MARS": {"distance": 228.0, "mass": 0.642, "inclination": 1.8, "orbital velocity": 24.1},
        "JUPITER": {"distance": 778.5, "mass": 1898, "inclination": 1.3, "orbital velocity": 13.1},
        "SATURN": {"distance": 1432.0, "mass": 568, "inclination": 2.5, "orbital velocity": 9.7},
        "URANUS": {"distance": 2867.0, "mass": 86.8, "inclination": 0.8, "orbital velocity": 6.8},
        "NEPTUNE": {"distance": 4515.0, "mass": 102, "inclination": 1.8, "orbital velocity": 5.4},
        "PLUTO": {"distance": 5906.4, "mass": 0.0130, "inclination": 17.2},
    }

    # RA DEC data on 21.01.25 at 00h00
    ra_dec = {
        'MERCURY': {'RA': (19, 20, 19.92), 'DEC': (-23, 27, 34.0), "distance": 57.96},
        'VENUS': {'RA': (23, 13, 30.71), 'DEC': (-4, 28, 29.0), "distance": 108.26},
        'EARTH': {'RA': (0, 0, 0.00), 'DEC': (0, 0, 0.0), "distance": 149.6},
        'MARS': {'RA': (7, 46, 32.47), 'DEC': (25, 34, 30.5), "distance": 228.0},
        'JUPITER': {'RA': (4, 39, 13.39), 'DEC': (21, 36, 5.4), "distance": 778.5},
        'SATURN': {'RA': (23, 11, 14.87), 'DEC': (-7, 20, 3.9), "distance": 1432.0},
        'URANUS': {'RA': (3, 22, 28.33), 'DEC': (18, 16, 9.7), "distance": 2867.0},
        'NEPTUNE': {'RA': (23, 52, 13.88), 'DEC': (-2, 13, 51.1), "distance": 4515.0},
    }

    # Calculate the initial coordinates of all the planets on 21.01.2024 at 00h00
    initial_coord = {}

    for planet, coord in ra_dec.items():
        # Extract RA, DEC, and distance
        ra_h, ra_min, ra_sec = coord['RA']
        dec_h, dec_min, dec_sec = coord['DEC']
        distance = coord['distance']

        # Convert RA and DEC to degrees
        ra_deg = ra_h * 15 + ra_min * 15 / 60 + ra_sec * 15 / 3600
        dec_deg = (abs(dec_h) + dec_min / 60 + dec_sec / 3600) * (-1 if dec_h < 0 else 1)

        # Convert degrees to radians
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)

        # Calculate Cartesian coordinates
        x = distance * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance * np.cos(dec_rad) * np.sin(ra_rad)
        z = distance * np.sin(dec_rad)

        initial_coord[planet] = {'x': x, 'y': y, 'z': z}

    earth_position = initial_coord['EARTH']

    # Adjust other planets' positions to be relative to the Sun
    for planet in initial_coord:
        if planet == 'EARTH':
            continue
        # Subtract Earth's position from the planet's position to shift to Sun-centered coordinates
        planet_position = initial_coord[planet]
        sun_centered_x = planet_position['x'] - earth_position['x']
        sun_centered_y = planet_position['y'] - earth_position['y']
        sun_centered_z = planet_position['z'] - earth_position['z']

        initial_coord[planet] = {'x': sun_centered_x, 'y': sun_centered_y, 'z': sun_centered_z}


    results = []
    planets_pos = np.zeros(shape=(10, 3), dtype=np.float64)
    planets_vel = np.zeros_like(planets_pos)
    planets_mass = np.zeros(10)
    i = 0
    for planet, data in planet_data.items():
        distance = data['distance']
        inclination = np.radians(data['inclination'])
        initial_position = initial_coord.get(planet, {'x': 0, 'y': 0, 'z': 0})

        # Use the initial positions as the starting coordinates
        x, y, z = initial_position['x'], initial_position['y'], initial_position['z']

        # Apply rotation matrix for inclination
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(inclination), -np.sin(inclination)],
            [0, np.sin(inclination), np.cos(inclination)]
        ])

        # Rotate the coordinates
        coords = np.array([x, y, z])
        rotated_coords = np.dot(rotation_matrix, coords)

        # Calculate velocity
        orb_vel = data.get('orbital velocity')

        # Skip planets without orbital velocity
        if orb_vel is None:
            continue
        theta = np.arctan2(y, x)
        v_x = -orb_vel * np.sin(theta)
        v_y = orb_vel * np.cos(theta)
        v_z = 0

        velocity = np.array([v_x, v_y, v_z])
        rotated_velocity = np.dot(rotation_matrix, velocity)

        # ADDED
        planets_pos[i, :] = rotated_coords
        planets_vel[i, :] = rotated_velocity
        planets_mass[i] = data["mass"]
        i += 1

        results.append({
            'Planet': planet,
            'Position X (kme6)': rotated_coords[0],
            'Position Y (kme6)': rotated_coords[1],
            'Position Z (kme6)': rotated_coords[2],
            'Velocity X (km/s)': rotated_velocity[0],
            'Velocity Y (km/s)': rotated_velocity[1],
            'Velocity Z (km/s)': rotated_velocity[2]
        })

    planets_mass[9] = 1988416.0
    return planets_pos, planets_vel, planets_mass
