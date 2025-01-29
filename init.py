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


def gmst(julian_date):
    # Calculates greenwich mean sidereal time from julian date

    jd_j2000 = 2451545.0
    T = (julian_date - jd_j2000) / 36525.0

    gmst_deg = (280.46061837 + 360.98564736629 * (julian_date - jd_j2000) +
                T**2 * (0.000387933 - T / 38710000.0)) % 360

    return np.radians(gmst_deg)


def teme_to_ecef(teme_coords, teme_velocities, julian_date):
    # Convert coordinates from TEME (True Equator Mean Equinox) to ECEF (Earth-Centered Earth-Fixed).

    GMST_rad = gmst(julian_date)[0]

    rotation_matrix = np.array([
        [np.cos(GMST_rad), np.sin(GMST_rad), 0],
        [-np.sin(GMST_rad), np.cos(GMST_rad), 0],
        [0, 0, 1]
    ])

    # Transform each satellite's TEME coordinates to ECEF
    ecef_coords = np.dot(teme_coords, rotation_matrix.T)
    ecef_velocities = np.dot(teme_velocities, rotation_matrix.T)

    return ecef_coords, ecef_velocities


def satellites(noDownload=False):
    # This function returns the positions and velocities of all the satellites
    # around earth on the date at the top of the file. It also return the index
    # of a geostationary sattelite named 'GOES 16'. This satellite like any
    # geostationary one is useful to determine where our launch site is. Instead
    # of keeping track of the rotation of the earth we can refer to the position
    # of this satellite to determine the normal vector of our launch site.

    if noDownload:
        data = np.load("satellites.npz")
        return data["pos"], data["vel"], data["goes_idx"]

    # Fetch TLE data and transform to SatrecArray
    try:
        response = requests.get(TLE_URL)
        response.raise_for_status()
        tle_data = response.text.splitlines()

        satellites = []
        names = []
        for i in range(0, len(tle_data), 3):
            if i + 2 < len(tle_data):
                name = tle_data[i].strip()
                line1 = tle_data[i + 1].strip()
                line2 = tle_data[i + 2].strip()
                try:
                    satrec = Satrec.twoline2rv(line1, line2)
                    satellites.append(satrec)
                    names.append(name)
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
    valid_sats = e.flatten() == 0
    pos = pos[valid_sats]
    vel = vel[valid_sats]
    valid_names = np.array(names)[valid_sats]
    try:
        goes_idx = np.where(valid_names == 'GOES 16')[0][0]
    except IndexError:
        print("Error: 'GOES 16' satellite not found in the filtered list.")
        goes_idx = None

    n_sats = len(pos)
    pos = np.reshape(pos, (n_sats, 3))
    vel = np.reshape(vel, (n_sats, 3))

    goes_idx = names.index('GOES 16')
    np.savez("satellites.npz", pos=pos, vel=vel, goes_idx=goes_idx)
    return pos, vel, goes_idx


# The planets in the list are in order starting with mercury. The last element
# of this list is the sun. Distance returned in (m), vel in (m/s) and mass in (kg)
def planets():
    planet_data = [[57.96, 7.0, 47.4],
                   [108.26, 3.4, 35.0],
                   [149.6, 0, 29.8],
                   [228.0, 1.8, 24.1],
                   [778.5, 1.3, 13.1],
                   [1432.0, 2.5, 9.7],
                   [2867.0, 0.8, 6.8],
                   [4515.0, 1.8, 5.4]]

    # RA DEC data on 21.01.25 at 00h00
    ra_dec = [[(19, 20, 19.92), (-23, 27, 34.0)],
              [(23, 13, 30.71), (-4, 28, 29.0)],
              [(0, 0, 0.00), (0, 0, 0.0)],
              [(7, 46, 32.47), (25, 34, 30.5)],
              [(4, 39, 13.39), (21, 36, 5.4)],
              [(23, 11, 14.87), (-7, 20, 3.9)],
              [(3, 22, 28.33), (18, 16, 9.7)],
              [(23, 52, 13.88), (-2, 13, 51.1)]]

    # 10^24 kg
    planets_mass = np.array([0.33, 4.87, 5.97, 0.642, 1898.0, 568.0, 86.8, 102.0, 988416.0])
    planets_pos = np.zeros(shape=(9, 3), dtype=np.float64)
    planets_vel = np.zeros_like(planets_pos)

    for planet in range(8):
        ra_h, ra_min, ra_sec = ra_dec[planet][0]
        dec_h, dec_min, dec_sec = ra_dec[planet][1]
        distance = planet_data[planet][0]

        ra = np.radians(ra_h * 15 + ra_min * 15 / 60 + ra_sec * 15 / 3600)
        dec = np.radians((abs(dec_h) + dec_min / 60 + dec_sec / 3600) * (-1 if dec_h < 0 else 1))

        # Calculate Cartesian coordinates
        planets_pos[planet, :] = np.array([distance * np.cos(dec) * np.cos(ra),
                                           distance * np.cos(dec) * np.sin(ra),
                                           distance * np.sin(dec)])

    # Adjust other planets' positions to be relative to the Sun
    earth_pos = np.array(planets_pos[2])
    planets_pos -= earth_pos
    planets_pos[2, :] = earth_pos
    planets_pos[8, :] += earth_pos

    for planet in range(8):
        distance = planet_data[planet][0]
        inclination = np.radians(planet_data[planet][1])

        # Calculate velocity
        theta = np.arctan2(planets_pos[planet, 1], planets_pos[planet, 0])
        vel = np.array([-planet_data[planet][2] * np.sin(theta),
                        planet_data[planet][2] * np.cos(theta),
                        0])

        # Apply rotation matrix for inclination
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(inclination), -np.sin(inclination)],
            [0, np.sin(inclination), np.cos(inclination)]
        ])

        planets_pos[planet] = np.dot(rotation_matrix, planets_pos[planet])
        planets_vel[planet] = np.dot(rotation_matrix, vel)

    #      (m)                (m/s)              (kg)
    return planets_pos * 1e9, planets_vel * 1e3, planets_mass * 1e24


def planets_original():
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


if __name__ == '__main__':
    pos, vel, goes_idx = satellites()
    print(goes_idx)
