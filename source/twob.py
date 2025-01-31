# This file contains the analytical solution for the two-body problem

import numpy as np

# Gravitational constant in km^3 kg^-1 s^-2
G = 6.67430e-20  # Adjusted for km and km/s

def two_body_analytical_update(pos1, vel1, mass1,
                               pos2, vel2, mass2,
                               dt):
    """
    Uses the analytical solution of the two-body problem (Keplerian orbit)
    to advance the current positions and velocities to the next time step.
    Returns (pos1_next, vel1_next, pos2_next, vel2_next).
    All positions are in km and velocities in km/s.
    """
    
    # 1) Calculate the gravitational parameter μ
    mu = G * (mass1 + mass2)
    
    # 2) Relative position and velocity
    R0 = pos2 - pos1  # Initial relative position (km)
    V0 = vel2 - vel1  # Initial relative velocity (km/s)
    
    # 3) Extract initial orbital elements
    
    # 3.1) Specific angular momentum vector h = R0 x V0
    h_vec = np.cross(R0, V0)
    h = np.linalg.norm(h_vec)
    
    # 3.2) Eccentricity vector e_vec
    # e_vec = (V0 x h_vec)/μ - R0/|R0|
    e_vec = (np.cross(V0, h_vec) / mu) - (R0 / np.linalg.norm(R0))
    e = np.linalg.norm(e_vec)
    
    # 3.3) Semi-major axis a
    # Total energy E = V^2/2 - mu/|R0|
    # a = - mu / (2 E)  (only valid for elliptical orbits e < 1)
    v2 = np.dot(V0, V0)
    r0 = np.linalg.norm(R0)
    energy = 0.5 * v2 - mu / r0
    a = - mu / (2.0 * energy)
    
    # 3.4) Inclination i
    # h_vec = (hx, hy, hz), i = arccos(hz / |h_vec|)
    i = np.arccos(h_vec[2] / h)
    
    # 3.5) Right ascension of the ascending node Omega
    # Node vector n_vec = k_hat x h_vec
    # k_hat = (0, 0, 1)
    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec)

    # Ω = arccos(nx / n), determine the quadrant based on ny
    if n != 0:
        Omega = np.arccos(n_vec[0]/n)
        if n_vec[1] < 0:
            Omega = 2.0*np.pi - Omega
    else:
        # Orbit lies in the reference plane, Omega is undefined set to 0
        Omega = 0.0
    
    # 3.6) Argument of periapsis omega
    # ω = arccos( n_vec · e_vec / (n*e) )
    # Determine the quadrant based on ez
    if n != 0 and e != 0:
        omega = np.arccos(np.dot(n_vec, e_vec)/(n*e))
        if e_vec[2] < 0:
            omega = 2.0*np.pi - omega
    else:
        omega = 0.0
    
    # 3.7) True anomaly ν0
    # ν0 = arccos( e_vec · R0 / (e * r0) )
    # Determine the quadrant based on (R0 · V0)
    if e != 0:
        cos_nu0 = np.dot(e_vec, R0) / (e * r0)
        # Clip to handle numerical errors beyond [-1,1]
        cos_nu0 = np.clip(cos_nu0, -1, 1)
        nu0 = np.arccos(cos_nu0)
        # If R0 · V0 < 0, then ν0 is in (π, 2π)
        if np.dot(R0, V0) < 0:
            nu0 = 2.0*np.pi - nu0
    else:
        nu0 = 0.0
    
    # 3.8) Mean anomaly M0
    # E0 = 2*arctan( sqrt((1-e)/(1+e))*tan(nu0/2) )
    # M0 = E0 - e sin(E0)
    E0 = 2.0*np.arctan( np.sqrt((1.0-e)/(1.0+e)) * np.tan(nu0*0.5) )
    M0 = E0 - e*np.sin(E0)
    # This is the mean anomaly at t=0
    
  
    # 4) Advance time to t = dt, calculate new M, E, ν, r
    # Mean motion
    n_mean = np.sqrt(mu / a**3)
    # M(dt) = M0 + n_mean * dt
    M = M0 + n_mean*dt
    
    # Solve Kepler's equation for eccentric anomaly E
    def kepler(M_val, e_val, tol=1e-12, max_iter=100):
        E_val = M_val  # Initial guess
        for _ in range(max_iter):
            f = E_val - e_val*np.sin(E_val) - M_val
            fp = 1.0 - e_val*np.cos(E_val)
            dE = -f / fp
            E_val += dE
            if abs(dE) < tol:
                break
        return E_val
    
    E = kepler(M, e)
    
    # True anomaly ν
    nu = 2.0*np.arctan2(
        np.sqrt(1+e)*np.sin(E*0.5),
        np.sqrt(1-e)*np.cos(E*0.5)
    )
    
    # Orbital radius r
    r = a*(1-e*e)/(1 + e*np.cos(nu))
    
    # 5) Calculate coordinates in the orbital plane (x_orb, y_orb)
    # Assume periapsis is along the x-axis:
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = 0.0

    # 6) Rotate (x_orb, y_orb, 0) to inertial coordinate system
    # Order: Rotate by Omega about z-axis, then by i about x-axis, then by omega about z-axis
    def rotation_z(alpha):
        return np.array([
            [ np.cos(alpha), -np.sin(alpha), 0],
            [ np.sin(alpha),  np.cos(alpha), 0],
            [           0,             0,    1]
        ])

    def rotation_x(alpha):
        return np.array([
            [1,           0,            0],
            [0, np.cos(alpha),-np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])
    
    # First rotate by omega around z-axis
    Rz_omega = rotation_z(omega)
    # Then rotate by inclination i around x-axis
    Rx_i = rotation_x(i)
    # Finally rotate by Omega around z-axis
    Rz_Omega = rotation_z(Omega)
    
    # Combine the rotation matrices
    # Order: pos_inertial = Rz_Omega * Rx_i * Rz_omega * pos_orb
    R_total = Rz_Omega @ Rx_i @ Rz_omega
    
    orb_pos = np.array([x_orb, y_orb, z_orb])
    R_new = R_total @ orb_pos  # Relative position in the inertial frame
    
    # 7) Distribute the relative position to each body based on the center of mass
    # r1 = -(m2 / (m1+m2)) R_new
    # r2 = +(m1 / (m1+m2)) R_new
    pos1_next = - (mass2/(mass1+mass2)) * R_new
    pos2_next = + (mass1/(mass1+mass2)) * R_new
    
    # 8) Calculate velocities based on the orbital equations
    # We use the vis-viva equation to determine the speed in the orbital plane
    # v^2 = mu (2/r - 1/a)
    v_mag = np.sqrt(mu*(2.0/r - 1.0/a))
    
    # Assuming the velocity is perpendicular to the radius vector in the orbital plane
    vx_orb = v_mag * (-np.sin(nu))
    vy_orb = v_mag * ( np.cos(nu))
    vz_orb = 0.0
    
    orb_vel = np.array([vx_orb, vy_orb, vz_orb])
    V_new = R_total @ orb_vel  # Velocity in the inertial frame
    
    # Distribute the velocity based on the center of mass
    vel1_next = - (mass2/(mass1+mass2)) * V_new
    vel2_next = + (mass1/(mass1+mass2)) * V_new
    
    return pos1_next, vel1_next, pos2_next, vel2_next
