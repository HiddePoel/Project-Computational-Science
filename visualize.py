import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from main import verlet_update


matplotlib.use('Qt5Agg')
plt.ion()


def update_plot(coords):
    scatters._offsets3d = (
        coords[:, 0],  # X-coordinates
        coords[:, 1],  # Y-coordinates
        coords[:, 2],  # Z-coordinates
    )
    plt.draw()
    plt.pause(0.01)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(-10, 10)
scatters = ax.scatter([], [], [], s=50)


plt.show()


points = np.array([[1, 1, 1], [20, 20, 0]], dtype=np.float64)
velocities = np.array([[0, 0, 0], [0, -10, 1]], dtype=np.float64)
masses = np.array([5000, 1], dtype=np.float64)


dt = 0.1
tts = 200
try:
    update_plot(points)
    for step in range(tts):
        points, velocities = verlet_update(points, velocities, masses, dt)
        # print(points)
        update_plot(points)
except KeyboardInterrupt:
    print("STOPPED")
