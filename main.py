import numpy as np


def init_solar():
    # Initialize position(3d), velocity(3d), and mass of celestial bodies not
    # including satellites. So sun, planets, moons.

    ...

    #return pos, vel, mass


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


def twobody_next_pos():
    ...


def twobody_update():
    ...


if __name__ == "__main__":
    ...
