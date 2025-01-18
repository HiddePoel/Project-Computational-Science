import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from main import verlet_update


matplotlib.use('Qt5Agg')
plt.ion()


# Function to update scatter plot in real time
def update_plot(coords):
    scatters._offsets3d = (
        coords[:, 0],  # X-coordinates
        coords[:, 1],  # Y-coordinates
        coords[:, 2],  # Z-coordinates
    )
    plt.draw()  # Redraw the updated plot
    plt.pause(0.01)  # Pause to allow real-time update


# Initialize the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Set axis limits (adjust as needed)
ax.set_xlim(0, 30)
ax.set_ylim(0, 30)
ax.set_zlim(0, 30)


# Create scatter plot (one scatter point per planet)
scatters = ax.scatter([], [], [], s=50)


plt.show()


points = np.array([[1, 3, 1], [5, 5, 3], [10, 10, 7]], dtype=np.float64)
velocities = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float64)
masses = np.array([10, 1, 4], dtype=np.float64)


dt = 0.1
tts = 100

try:
    update_plot(points)
    for step in range(tts):
        points, velocities = verlet_update(points, velocities, masses, dt)
        # print(points)
        update_plot(points)
except KeyboardInterrupt:
    print("STOPPED")
