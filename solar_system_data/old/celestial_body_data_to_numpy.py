import os
import numpy as np

"""
Summary of the data format for celestial body data files according to the JPL Horizons system:

TIME:
- Times before 1962 are in UT1 (mean-solar time); after 1962, they are in UTC (modern civil time).
- Leap seconds keep UTC within 0.9 seconds of UT1.
- Dates reference UT on Earth, regardless of local gravitational effects.

CALENDAR SYSTEM:
- Mixed calendar mode: Gregorian after 1582-Oct-15, Julian before.
- "n.a." indicates unavailable quantities.

'R.A._____(ICRF)_____DEC':
- Right Ascension (RA) and Declination (DEC) in ICRF frame, corrected for light-time delay.
- Units: RA in HH MM SS.ff and DEC in ° ' ".

'APmag   S-brt':
- Apparent magnitude (brightness) and surface brightness.
- "n.a." appears if data is outside valid observational ranges.
- Units: magnitudes.

'delta      deldot':
- Delta: Distance from the observer in AU.
- Deldot: Rate of distance change; positive = moving away, negative = approaching.
- Units: AU and km/s.

'S-O-T /r':
- Sun-Observer-Target angle (solar elongation).
- "/T" indicates the target trails the Sun; "/L" indicates it leads.
- Units: degrees.

'S-T-O':
- Sun-Target-Observer angle, with light-time corrections.
- Units: degrees.

'Sky_motion  Sky_mot_PA  RelVel-ANG':
- Sky motion rate and position angle of motion.
- RelVel-ANG shows the relative velocity angle:
  -90° = moving toward observer, +90° = moving away.
- Units: arcseconds/minute, degrees.

'Lun_Sky_Brt  sky_SNR':
- Sky brightness due to moonlight and signal-to-noise ratio (SNR).
- "n.a." appears if conditions like the Moon's position or Sun's twilight status are unmet.
- Units: visual magnitudes/arcsecond² and unitless SNR.
"""

DEBUG = True

def read_data(file_path: str) -> np.ndarray:
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return np.empty((0, 6), dtype=np.float64)

    # Find the start and end indices of the data block
    start_index, end_index = None, None
    for i, line in enumerate(lines):
        if line.strip() == "$$SOE":
            start_index = i
        elif line.strip() == "$$EOE":
            end_index = i
            break

    if start_index is None:
        print(f"Error: '$$SOE' marker not found in '{file_path}'.")
        return np.empty((0, 6), dtype=np.float64)
    if end_index is None:
        print(f"Error: '$$EOE' marker not found in '{file_path}'.")
        return np.empty((0, 6), dtype=np.float64)

    # Extract data lines
    data_lines = lines[start_index + 1 : end_index]

    # Prepare to store the required fields as numeric columns
    data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < 15:
            print(f"Skipping incomplete line: {line.strip()}")
            continue
        try:
            # Parse RA (hours, minutes, seconds)
            ra_h = float(parts[2])
            ra_m = float(parts[3])
            ra_s = float(parts[4])

            # Parse DEC (degrees, minutes, seconds)
            dec_d = float(parts[5])
            dec_m = float(parts[6])
            dec_s = float(parts[7])

            # Parse delta (distance)
            delta = float(parts[10])

            # Store all as numeric values
            data.append([ra_h, ra_m, ra_s, dec_d, dec_m, dec_s, delta])
        except ValueError:
            print(f"Skipping malformed line: {line.strip()}")
            continue

    return np.array(data, dtype=np.float64)


def calculate_xyz_coordinates(data: np.ndarray) -> np.ndarray:
    # Debug print to verify the shape and sample data
    if DEBUG:
        print(f"Debug: Data shape = {data.shape}, Sample row = {data[0]}")

    try:
        # Extract delta (distance in AU)
        delta = data[:, 6]  # The 7th column is delta

        # Convert RA (hours, minutes, seconds) to degrees
        ra_deg = (data[:, 0] + data[:, 1] / 60 + data[:, 2] / 3600) * 15

        # Convert DEC (degrees, minutes, seconds) to degrees
        dec_deg = np.sign(data[:, 3]) * (np.abs(data[:, 3]) + data[:, 4] / 60 + data[:, 5] / 3600)

        # Convert RA and DEC to radians
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
    except IndexError as e:
        print(f"Error: Data format mismatch. Ensure input contains 7 columns. Details: {e}")
        return np.array([])
    except ValueError as e:
        print(f"Error: Non-numeric values detected in data. Details: {e}")
        return np.array([])

    # Calculate Cartesian coordinates
    x = delta * np.cos(dec_rad) * np.cos(ra_rad)
    y = delta * np.cos(dec_rad) * np.sin(ra_rad)
    z = delta * np.sin(dec_rad)

    # Combine x, y, z into a single NumPy array
    return np.column_stack((x, y, z))


celestial_body_names = [
    "sun",
    "mercury",
    "venus",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
]

filenames = [body_name + ".txt" for body_name in celestial_body_names]

script_dir = os.path.dirname(os.path.abspath(__file__))

for filename in filenames:
    data = read_data(script_dir + "/" + filename)
    coordinates = calculate_xyz_coordinates(data)
    print(f"saving {filename.split('.')[0]}_coordinates.npy")
    np.save(script_dir + "/" + filename.split(".")[0] + "_coordinates.npy", coordinates)
    if DEBUG:
        print(f"{coordinates[:100]}")
        DEBUG = False
