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


def read_data(file_path: str) -> np.ndarray:
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the index of the "$$SOE" marker
    start_index = 0
    for i, line in enumerate(lines):
        if line.strip() == "$$SOE":
            start_index = i + 1
            break

    # Extract data lines after the "$$SOE" marker
    data_lines = []
    for line in lines[start_index:]:
        if line.strip() == "$$EOE":
            break
        data_lines.append(line)

    # Prepare a list to hold the parsed data
    data = []

    for line in data_lines:
        # print(f"Bug0:    {line}")
        parts = line.split()
        if len(parts) < 15:
            continue
        date_time = parts[0] + " " + parts[1]
        ra = parts[2] + " " + parts[3] + " " + parts[4]
        dec = parts[5] + " " + parts[6] + " " + parts[7]
        apmag = float(parts[8])
        s_brt = float(parts[9])
        delta = float(parts[10])
        deldot = float(parts[11])
        s_o_t = float(parts[12])
        # print(f"Bug:    {parts[13]}")
        s_t_o = float(parts[14])
        sky_motion = float(parts[15])
        sky_mot_pa = float(parts[16])
        relvel_ang = float(parts[17])
        lun_sky_brt = parts[18]
        sky_snr = parts[19]

        data.append(
            [
                date_time,
                ra,
                dec,
                apmag,
                s_brt,
                delta,
                deldot,
                s_o_t,
                s_t_o,
                sky_motion,
                sky_mot_pa,
                relvel_ang,
                lun_sky_brt,
                sky_snr,
            ]
        )

    # Convert the list to a NumPy array
    data_array = np.array(data, dtype=object)

    return data_array


def calculate_xyz_coordinates(data: np.ndarray) -> np.ndarray:
    # Extract the distance from the observer (in AU)
    delta = data[:, 5].astype(float)

    # Extract the Sun-Observer-Target angle (solar elongation) in degrees
    s_o_t = data[:, 7].astype(float)

    # Calculate the x, y, z coordinates in the observer's frame
    x = delta * np.cos(np.radians(s_o_t))
    y = delta * np.sin(np.radians(s_o_t))
    z = np.zeros_like(x)

    # Combine the x, y, z coordinates into a single NumPy array
    xyz_coordinates = np.column_stack((x, y, z))

    return xyz_coordinates


filenames = ["sun.txt", "mercury.txt", "venus.txt", "mars.txt", "jupiter.txt", "saturn.txt", "uranus.txt", "neptune.txt"]

for filename in filenames:
    data = read_data(filename)
    ...

