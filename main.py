import numpy as np


def init_solar():
    # Initialize position(3d), velocity(3d), and mass of celestial bodies not
    # including satellites. So sun, planets, moons.

    ...

    # return pos, vel, mass


def init_satellites():
    # same as above.
    ...


def force(pos_a, pos_b, m_a, m_b):
    # G = 6.67408e-11
    G = 1

    # Force on point A due to point B
    F = G * m_a * m_b / np.linalg.norm(pos_b - pos_a)**3 * (pos_b - pos_a)
    return F


def force_all(idx, pos, mass):
    # calculate the force on body with index 'idx', due to all
    # the other bodies (with indexes =/= 'idx'). 'pos' and 'mass' are arrays.
    mask = np.ones_like(mass, dtype=bool)
    mask[idx] = False

    pos_B = pos[mask]
    m_B = mass[mask]

    pos_a = pos[idx]
    m_a = mass[idx]

    F = np.array((0.0, 0.0, 0.0))
    for i in range(len(pos_B)):
        F += force(pos_a, pos_B[i], m_a, m_B[i])

    return F


def accel(pos_a, pos_b, m_a, m_b):
    return force(pos_a, pos_b, m_a, m_b) / m_a


def accel_all(idx, pos, mass):
    mask = np.ones_like(mass, dtype=bool)
    mask[idx] = False

    pos_B = pos[mask]
    m_B = mass[mask]

    pos_a = pos[idx]
    m_a = mass[idx]

    a = 0.0
    for i in range(len(pos_B)):
        a += accel(pos_a, pos_B[i], m_a, m_B[i])
    return a


def verlet_update(poss, vels, masses, dt):
    N = len(poss)

    accs = np.zeros_like(vels, dtype=np.float64)
    for idx in range(N):
        accs[idx] = accel_all(idx, poss, masses)

    pos_next = np.zeros_like(poss, dtype=np.float64)
    for idx in range(N):
        pos_next[idx] = poss[idx] + vels[idx] * dt + 0.5 * accs[idx] * dt**2

    acc_next = np.zeros_like(accs)
    for idx in range(N):
        acc_next[idx] = accel_all(idx, pos_next, masses)

    vel_next = np.zeros_like(vels, dtype=np.float64)
    for idx in range(N):
        vel_next[idx] = vels[idx] + (accs[idx] + acc_next[idx]) / 2 * dt

    return pos_next, vel_next


# used for earth sim
def twobody_next_pos():
    ...


# used for earth sim
def twobody_update():
    ...


def sat_opening(pos_sat, pos_launch, normal_launch):
    # calculates how large the opening is above launch location. We need to
    # somehow keep track of earth's orientation idk how. we can start with
    # the math to actually calculate the opening first

    # pos_launch = 3d pos
    # normal_launch = vector perpendicular to earth surface at pos_launch.

    # return the radius of the largest cilinder we can make in the atmosphere
    # in the direction of 'normal_launch' that doesn't contain a sattelite.
    ...


def vis_earth(current_pos):
    ...


def vis_solar(current_pos):
    ...


def main_solar():
    pos, vel, m = init_solar()

    # time to sim
    tts = 100
    dt = 0.1

    # acc = np.zeros_like(pos, dtype=np.float64)
    # for idx in range(len(pos)):
    #     acc[idx] = accel_all(idx, pos, m)

    pos_next, vel_next = verlet_update(pos, vel, m, dt)
    for step in range(tts - 1 // dt):
        # vis_solar(pos)
        pos, vel = verlet_update(pos_next, vel_next, m, dt)


def main_earth():
    pos, vel, m = init_satellites()

    # time to sim
    tts = 100
    dt = 0.1

    opening_thresh = 10
    launcht_candidates = []

    for step in range(tts // dt):
        vis_earth(pos)
        twobody_update()
        if opening_thresh <= sat_opening():
            # snapshot current time and position of satellites and earth
            # add this to launcht_candidates or somewhere else.
            ...


if __name__ == "__main__":
    ...
