
import numpy as np
from numba import njit

G = 6.67430e-20  # Gravitational constant in km^3 kg^-1 s^-2
MASS1 = 5.972e24
MASS2 = 500.0
MU = G * (MASS1 + MASS2)


def kepler_solver(M, e, tol=1e-12, max_iter=100):
    E = M  # Initial guess
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E

@njit
def rotation_matrix(Omega, i, omega):
    cos_O, sin_O = np.cos(Omega), np.sin(Omega)
    cos_i, sin_i = np.cos(i), np.sin(i)
    cos_w, sin_w = np.cos(omega), np.sin(omega)

    return np.array([
        [cos_O * cos_w - sin_O * sin_w * cos_i, -cos_O * sin_w - sin_O * cos_w * cos_i, sin_O * sin_i],
        [sin_O * cos_w + cos_O * sin_w * cos_i, -sin_O * sin_w + cos_O * cos_w * cos_i, -cos_O * sin_i],
        [sin_w * sin_i, cos_w * sin_i, cos_i]
    ])


def twobo(pos2, vel2, mass2, dt):
    mass1 = 5.972e24
    mu = G * (mass1 + mass2)
    R0 = pos2  # Relative position
    V0 = vel2  # Relative velocity
    r0_inv = 1 / np.linalg.norm(R0)

    h_vec = np.cross(R0, V0)
    h = np.linalg.norm(h_vec)
    e_vec = np.cross(V0, h_vec) / mu - R0 * r0_inv
    e = np.linalg.norm(e_vec)
    v2 = np.dot(V0, V0)
    energy = 0.5 * v2 - mu * r0_inv
    a = -mu / (2 * energy)
    i = np.arccos(h_vec[2] / h)

    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec)
    Omega = np.arccos(n_vec[0] / n) if n != 0 else 0.0
    if n != 0 and n_vec[1] < 0:
        Omega = 2.0 * np.pi - Omega

    omega = np.arccos(np.dot(n_vec, e_vec) / (n * e)) if n != 0 and e != 0 else 0.0
    if e_vec[2] < 0:
        omega = 2.0 * np.pi - omega

    cos_nu0 = np.clip(np.dot(e_vec, R0) / (e * np.linalg.norm(R0)), -1, 1)
    nu0 = np.arccos(cos_nu0)
    if np.dot(R0, V0) < 0:
        nu0 = 2.0 * np.pi - nu0

    E0 = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu0 / 2), np.sqrt(1 + e) * np.cos(nu0 / 2))
    M0 = E0 - e * np.sin(E0)
    n_mean = np.sqrt(mu / a**3)
    M = M0 + n_mean * dt
    E = kepler_solver(M, e)

    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e**2) / (1 + e * np.cos(nu))

    x_orb, y_orb = r * np.cos(nu), r * np.sin(nu)
    R_total = rotation_matrix(Omega, i, omega)
    R_new = R_total @ np.array([x_orb, y_orb, 0])

    pos2_next = + (mass1 / (mass1 + mass2)) * R_new

    v_mag = np.sqrt(mu * (2.0 / r - 1.0 / a))
    vx_orb, vy_orb = -v_mag * np.sin(nu), v_mag * np.cos(nu)
    V_new = R_total @ np.array([vx_orb, vy_orb, 0])

    vel2_next = + (mass1 / (mass1 + mass2)) * V_new

    return pos2_next, vel2_next


def kepler_solver_vectorized(M, e, tol=1e-12, max_iter=100):
    E = M  # Initial guess
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def rotation_matrix_vectorized(Omega, i, omega):
    cos_O, sin_O = np.cos(Omega).squeeze(), np.sin(Omega).squeeze()
    cos_i, sin_i = np.cos(i).squeeze(), np.sin(i).squeeze()
    cos_w, sin_w = np.cos(omega).squeeze(), np.sin(omega).squeeze()

    R = np.empty((Omega.shape[0], 3, 3))
    R[:, 0, 0] = cos_O * cos_w - sin_O * sin_w * cos_i
    R[:, 0, 1] = -cos_O * sin_w - sin_O * cos_w * cos_i
    R[:, 0, 2] = sin_O * sin_i
    R[:, 1, 0] = sin_O * cos_w + cos_O * sin_w * cos_i
    R[:, 1, 1] = -sin_O * sin_w + cos_O * cos_w * cos_i
    R[:, 1, 2] = -cos_O * sin_i
    R[:, 2, 0] = sin_w * sin_i
    R[:, 2, 1] = cos_w * sin_i
    R[:, 2, 2] = cos_i

    return R


def twobo_vectorized(pos2, vel2, dt):
    # Had this returned accurate values, it would have seen about a 10x speedup according to tests.
    r0_inv = 1 / np.linalg.norm(pos2, axis=1, keepdims=True)

    h_vec = np.cross(pos2, vel2)
    h = np.linalg.norm(h_vec, axis=1, keepdims=True)
    e_vec = np.cross(vel2, h_vec) / MU - pos2 * r0_inv
    e = np.linalg.norm(e_vec, axis=1, keepdims=True)
    v2 = np.sum(vel2 ** 2, axis=1, keepdims=True)
    energy = 0.5 * v2 - MU * r0_inv
    a = -MU / (2 * energy)
    i = np.arccos(h_vec[:, 2:3] / h)

    k_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec, axis=1, keepdims=True)
    Omega = np.where(n != 0, np.arccos(n_vec[:, 0:1] / n), 0.0)
    Omega[n_vec[:, 1:2] < 0] = 2.0 * np.pi - Omega[n_vec[:, 1:2] < 0]

    omega = np.where((n != 0) & (e != 0), np.arccos(np.einsum('ij,ij->i', n_vec, e_vec)[:, np.newaxis] / (n * e)), 0.0)
    omega[e_vec[:, 2:3] < 0] = 2.0 * np.pi - omega[e_vec[:, 2:3] < 0]

    cos_nu0 = np.einsum('ij,ij->i', e_vec, pos2)[:, np.newaxis] / (e * np.linalg.norm(pos2, axis=1, keepdims=True))
    nu0 = np.arccos(np.clip(cos_nu0, -1, 1))
    nu0[np.einsum('ij,ij->i', pos2, vel2)[:, np.newaxis] < 0] = 2.0 * np.pi - nu0[np.einsum('ij,ij->i', pos2, vel2)[:, np.newaxis] < 0]

    E0 = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu0 / 2), np.sqrt(1 + e) * np.cos(nu0 / 2))
    M0 = E0 - e * np.sin(E0)
    n_mean = np.sqrt(MU / a**3)
    M = M0 + n_mean * dt
    E = kepler_solver_vectorized(M, e)

    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e**2) / (1 + e * np.cos(nu))

    x_orb, y_orb = r * np.cos(nu), r * np.sin(nu)
    R_total = rotation_matrix_vectorized(Omega, i, omega)
    R_new = np.einsum('nij,nkj->nik', R_total, np.stack([x_orb, y_orb, np.zeros_like(x_orb)], axis=-1))
    R_new = R_new.squeeze()
    pos2_next = (MASS1 / (MASS1 + MASS2)) * R_new

    v_mag = np.sqrt(MU * (2.0 / r - 1.0 / a))
    vx_orb, vy_orb = -v_mag * np.sin(nu), v_mag * np.cos(nu)
    V_new = np.einsum('nij,nkj->nik', R_total, np.stack([vx_orb, vy_orb, np.zeros_like(vx_orb)], axis=-1))
    V_new = V_new.squeeze()
    vel2_next = (MASS1 / (MASS1 + MASS2)) * V_new

    return pos2_next, vel2_next
