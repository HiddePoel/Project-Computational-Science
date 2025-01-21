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
jd, fr = jday(INIT_YEAR, INIT_MONTH, INIT_DAY, INIT_HOUR, INIT_MINUTE, INIT_SECOND)
jd = np.array([jd])
fr = np.array([fr])

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
            except Exception as e:
                print(f"Error parsing TLE for satellite '{name}': {e}")

except requests.exceptions.RequestException as e:
    print(f"Error fetching TLE data: {e}")

N = len(satellites)
print("Number of satellites: ", N)
oe = np.zeros(shape=(N, 6), dtype=np.float64)
for s in range(N):
    sat = satellites[s]
    oe[s, :] = np.array([sat.ecco, sat.a, sat.inclo, sat.nodeo, sat.argpo, sat.mo])

print(oe[0])

satarr = SatrecArray(satellites)
e, r, v = satarr.sgp4(jd, fr)
