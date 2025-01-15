import numpy as np


def init_solar():
    # Initialize position(3d), velocity(3d), and mass of celestial bodies not
    # including satellites. So sun, planets, moons.

    ...

    # return pos, vel, mass


def init_satellites():
    # same as above.
    ...


def force(idx, pos, mass):
    # calculate the force on body with index 'idx', due to all
    # the other bodies (with indexes =/= 'idx'). 'pos' and 'mass' are arrays.
    ...
    # return f


def accel(idx, pos, mass):
    ...
    # return a


def verlet_next_pos(pos_t, pos_mindt, accel_t, dt):
    # part of verlet. arguments: pos(t), pos(t - dt), accel(t), dt
    # this func returns pos(t + dt), next pos.

    ...

    # return pos_tplusdt


def verlet_next_vel(vel_t, accel_t, accel_tplusdt, dt):
    # part of verlet.

    ...

    # return vel_tplusdt


def verlet_update():
    # wip
    ...


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

    for step in range(tts // dt):
        vis_solar(pos)
        verlet_update()


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
