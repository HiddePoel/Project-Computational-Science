from main import update_satellite, sat_opening, get_launch_site
from init import satellites
import numpy as np
import matplotlib.pyplot as plt
from optimized_twob import twobo, twobo_multi

dt = 300
steps = 10000
opening_sizes = np.zeros(steps)
m = np.array([0.0, 0.0, 5.972e24])

sats_pos, sats_vel, goes_idx = satellites(noDownload=True)
avg_velocities = np.zeros(len(sats_pos))
# mul_pos, mul_vel = sats_pos.copy(), sats_vel.copy()
# mul_opening_sizes = np.zeros(steps)

for i in range(steps):
    sats_pos, sats_vel = update_satellite(sats_pos, sats_vel, m, dt)
    # for sat in range(len(sats_pos)):
    #     sats_pos[sat], sats_vel[sat] = twobo(sats_pos[sat], sats_vel[sat], m, dt)
    pos_launch, point_above = get_launch_site(0, sats_pos, goes_idx)
    launch_normal = point_above - pos_launch

    opening_sizes[i] = sat_opening(sats_pos, pos_launch, launch_normal, goes_idx)
    avg_velocities[i] = np.mean(np.linalg.norm(sats_vel, axis=1))

np.savez("opening_sizes_every_300_sec_1000.npy", opening_sizes=opening_sizes)
np.savez("avg_velocities_every_300_sec_1000.npy", avg_velocities=avg_velocities)

plt.plot(opening_sizes)
plt.xlabel("Time (300 seconds)")
plt.ylabel("Opening radius (km)")
plt.title("Opening radius over time")
plt.show()

plt.plot(avg_velocities)
plt.xlabel("Time (300 seconds)")
plt.ylabel("Average velocity (km/s)")
plt.title("Average velocity of satellites over time")
plt.show()


# for i in range(steps):
#     print(mul_pos.shape)

#     sats_pos, sats_vel = update_satellite(sats_pos, sats_vel, m, dt)
#     # for sat in range(len(sats_pos)):
#     #     mul_pos[sat], mul_vel[sat] = twobo(mul_pos[sat], mul_vel[sat], 500, dt)
#     mul_pos, mul_vel = twobo_multi(mul_pos, mul_vel, dt)

#     pos_launch, point_above = get_launch_site(0, sats_pos, goes_idx)
#     launch_normal = point_above - pos_launch
#     opening_sizes[i] = sat_opening(sats_pos, pos_launch, launch_normal, goes_idx)

#     mul_launch, mul_above = get_launch_site(0, mul_pos, goes_idx)
#     mul_normal = mul_above - mul_launch
#     mul_opening_sizes[i] = sat_opening(mul_pos, mul_launch, mul_normal, goes_idx)


# print(opening_sizes)
# print(mul_opening_sizes)
# print(opening_sizes - mul_opening_sizes)

