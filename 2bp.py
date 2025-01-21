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


def kepler(mo, ecco, tolerance=1e-6, max_iter=100):
    # Returns eccentric anomaly.
    E = mo
    for _ in range(max_iter):
        dE = (mo - (E - ecco * np.sin(E))) / (1 - ecco * np.cos(E))
        E += dE
        if abs(dE) < tolerance:
            break

    return E


def position(t, oe, t0, m, G=1.0):
    ecco = oe[0]
    a = oe[1]
    inclo = oe[2]
    nodeo = oe[3]
    argpo = oe[4]
    mo = oe[5]

    n = np.sqrt(G * m / a ** 3)
    M = mo + n * (t - t0)
    E = kepler(M, ecco)

    nu = 2 * np.arctan(np.sqrt((1 + ecco) / (1 - ecco)) * np.tan(E / 2))
    r = a * (1 - ecco ** 2) / (1 + ecco * np.cos(nu))

    x_orb = r * (np.cos(nodeo) * np.cos(argpo + nu) - np.sin(nodeo) * np.sin(argpo + nu) * np.cos(inclo))
    y_orb = r * (np.sin(nodeo) * np.cos(argpo + nu) + np.cos(nodeo) * np.sin(argpo + nu) * np.cos(inclo))
    z_orb = r * np.sin(inclo)

    return np.array([x_orb, y_orb, z_orb])


def validate():
    sats = satellites[:2]
    oe = np.zeros(shape=(2, 6), dtype=np.float64)
    for s in range(2):
        sat = sats[s]
        oe[s, :] = np.array([sat.ecco, sat.a, sat.inclo, sat.nodeo, sat.argpo, sat.mo])

    cdata = [sat.sgp4(jd, fr) for sat in sats]
    pdata = [position(jd, oe[i], jd, 5.972e24, 6.67430e-11) for i in range(2)]
    print(cdata)
    print(pdata)
    print(cdata - pdata)


validate()

# satarr = SatrecArray(satellites)
# e, r, v = satarr.sgp4(jd, fr)
