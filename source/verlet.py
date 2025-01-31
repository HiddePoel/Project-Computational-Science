import numpy as np


def force(pos_a, pos_b, m_a, m_b):
    # G = 6.67408e-11
    G = 1.0

    # Force on point A due to point B
    F = G * m_a * m_b / np.linalg.norm(pos_b - pos_a)**3 * (pos_b - pos_a)
    return F


def pairwise_forces(pos, masses, G=1.0):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    distances_inv3 = np.where(distances > 0, distances**-3, 0)
    # print("FORCES")
    # print(distances_inv3)
    # print("FORCES END")
    forces = G * diff * (distances_inv3[:, :, np.newaxis] * (masses[:, np.newaxis] * masses[np.newaxis, :])[:, :, np.newaxis])

    net_forces = np.sum(forces, axis=1) * -1
    return net_forces


def update(pos, vel, masses, dt, G=1.0):
    acc = pairwise_forces(pos, masses, G) / masses[:, np.newaxis]
    pos_next = pos + vel * dt + 0.5 * acc * dt**2
    acc_next = pairwise_forces(pos_next, masses, G) / masses[:, np.newaxis]
    vel_next = vel + 0.5 * (acc + acc_next) * dt

    return pos_next, vel_next
