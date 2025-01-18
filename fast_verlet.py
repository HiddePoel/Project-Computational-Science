import numpy as np
import main


points = np.random.uniform(-100, 100, size=(10, 3))
vels = np.random.uniform(-10, 10, size=(10, 3))
masses = np.random.uniform(1, 100, size=10)
print(masses[np.newaxis, :, np.newaxis])


def pairwise_distances(pos):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    return distances, diff


# dist, diff = pairwise_distances(points)
# print(diff[0])
# diff2 = np.zeros_like(diff[0])
# for i in range(len(points)):
#     diff2[i] = points[0] - points[i]
# print(diff2)


def pairwise_forces(pos, masses, G=1.0):
    distances, diff = pairwise_distances(pos)
    distances_inv3 = np.where(distances > 0, distances**-3, 0)
    forces = G * diff * (distances_inv3[:, :, np.newaxis] * (masses[:, np.newaxis] * masses[np.newaxis, :])[:, :, np.newaxis])
    net_forces = np.sum(forces, axis=1)
    return net_forces


pw_f = pairwise_forces(points, masses)
print(pw_f[0])
print()

f = main.force_all(0, points, masses)
print(f)


def fast_verlet_update(pos, vel, masses, dt, G=1.0):
    acc = pairwise_forces(pos, masses, G) / masses[:, np.newaxis]
    pos_next = pos + vel * dt + 0.5 * acc * dt**2
    acc_next = pairwise_forces(pos_next, masses, G) / masses[:, np.newaxis]
    vel_next = vel + 0.5 * (acc + acc_next) * dt

    return pos_next, vel_next

