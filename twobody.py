import numpy as np

"""
# used for earth sim
def twobody_update(pos1, pos2, vel1, vel2, mass1, mass2, dt):

    # Compute gravitational force on body 1 due to body 2
    F = force(pos1, pos2, mass1, mass2)
    
    # Compute accelerations (Newton's third law)
    a1 = F / mass1
    a2 = -F / mass2  
    
    # Update positions
    pos1_next = pos1 + vel1 * dt + 0.5 * a1 * dt**2
    pos2_next = pos2 + vel2 * dt + 0.5 * a2 * dt**2
    
    # Compute forces at next positions
    F_next = force(pos1_next, pos2_next, mass1, mass2)
    
    # Compute new accelerations
    a1_next = F_next / mass1
    a2_next = -F_next / mass2
    
    # Update velocities
    vel1_next = vel1 + 0.5 * (a1 + a1_next) * dt
    vel2_next = vel2 + 0.5 * (a2 + a2_next) * dt
    
    return pos1_next, pos2_next, vel1_next, vel2_next
"""

def kepler_solver(M, e, tolerance=1e-6, max_iter=100):
    """
    Solves Kepler's equation for the eccentric anomaly E given mean anomaly M and eccentricity e.
    
    Parameters:
    - M (float): Mean anomaly (radians).
    - e (float): Orbital eccentricity.
    - tolerance (float): Convergence tolerance.
    - max_iter (int): Maximum number of iterations.

    Returns:
    - E (float): Eccentric anomaly (radians).
    """
    E = M  # Initial guess
    for _ in range(max_iter):
        dE = (M - (E - e * np.sin(E))) / (1 - e * np.cos(E))
        E += dE
        if abs(dE) < tolerance:
            break
    return E

def satellite_position(t, oe, t0, m_central, G=6.67430e-11):
    """
    Computes the 3D position of a satellite using orbital elements and time.

    Parameters:
    - t (float): Current time (seconds).
    - oe (array): Orbital elements [e, a, i, Ω, ω, M0] (eccentricity, semi-major axis,
                  inclination, RAAN, argument of perigee, mean anomaly at t0).
    - t0 (float): Reference time (seconds).
    - m_central (float): Mass of the central body (e.g., Earth).
    - G (float): Gravitational constant.

    Returns:
    - pos (array): 3D position of the satellite (x, y, z) in meters.
    """
    e, a, i, raan, argp, M0 = oe

    # Mean motion
    n = np.sqrt(G * m_central / a**3)

    # Mean anomaly at time t
    M = M0 + n * (t - t0)

    # Solve Kepler's equation for eccentric anomaly E
    E = kepler_solver(M, e)

    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )

    # Radial distance
    r = a * (1 - e**2) / (1 + e * np.cos(nu))

    # Position in the orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Check for edge case: no inclination, RAAN, or argument of perigee
    if np.isclose(i, 0) and np.isclose(raan, 0) and np.isclose(argp, 0):
        return np.array([x_orb, y_orb, 0])

    # General case: apply transformations for inclination, RAAN, and argument of perigee
    x = (
        x_orb * (np.cos(raan) * np.cos(argp) - np.sin(raan) * np.sin(argp) * np.cos(i))
        - y_orb * (np.cos(raan) * np.sin(argp) + np.sin(raan) * np.cos(argp) * np.cos(i))
    )
    y = (
        x_orb * (np.sin(raan) * np.cos(argp) + np.cos(raan) * np.sin(argp) * np.cos(i))
        + y_orb * (np.cos(raan) * np.cos(argp) - np.sin(raan) * np.sin(argp) * np.cos(i))
    )
    z = (x_orb * np.sin(argp) + y_orb * np.cos(argp)) * np.sin(i)

    return np.array([x, y, z])



def twobody_update(t, oe_list, t0, m_central, G=6.67430e-11):
    """
    Updates the positions of satellites in a two-body system analytically.

    Parameters:
    - t (float): Current time (seconds).
    - oe_list (list of arrays): List of orbital elements for each satellite.
    - t0 (float): Reference time (seconds).
    - m_central (float): Mass of the central body (e.g., Earth).
    - G (float): Gravitational constant.

    Returns:
    - positions (array): 3D positions of satellites (N x 3) in meters.
    """
    positions = []
    for oe in oe_list:
        pos = satellite_position(t, oe, t0, m_central, G)
        positions.append(pos)
    return np.array(positions)

# Example usage:
# Orbital elements for satellites: [e, a, i, RAAN, argument of perigee, M0]
satellite_oe = [
    [0.01, 7000e3, np.radians(30), np.radians(40), np.radians(60), np.radians(0)],
    [0.02, 7100e3, np.radians(50), np.radians(80), np.radians(90), np.radians(10)]
]

# Central body (Earth) parameters
earth_mass = 5.972e24  # kg

# Compute satellite positions at t = 3600 seconds (1 hour after t0)
positions = twobody_update(t=3600, oe_list=satellite_oe, t0=0, m_central=earth_mass)
print("Satellite positions:", positions)
