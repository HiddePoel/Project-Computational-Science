import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from twobody import kepler_solver, satellite_position, twobody_update


#test cases
class TestTwoBody(unittest.TestCase):
    def test_kepler_solver(self):
        # Test for a circular orbit (eccentricity = 0)
        M = np.radians(30)
        e = 0
        E = kepler_solver(M, e)
        self.assertAlmostEqual(E, M, places=6)

    def test_satellite_position(self):
        # Simple case: circular orbit with no inclination or argument of perigee
        t = 3600  # 1 hour
        oe = [0.0, 7000e3, np.radians(0), np.radians(0), np.radians(0), np.radians(0)]
        t0 = 0
        m_central = 5.972e24  # Earth's mass
        pos = satellite_position(t, oe, t0, m_central)
        self.assertTrue(np.allclose(pos, [7000e3, 0, 0], atol=1e3))  # Rough position

        

    def test_twobody_update(self):
        # Multiple satellites
        t = 3600
        satellite_oe = [
            [0.01, 7000e3, np.radians(30), np.radians(40), np.radians(60), np.radians(0)],
            [0.02, 7100e3, np.radians(50), np.radians(80), np.radians(90), np.radians(10)],
        ]
        t0 = 0
        m_central = 5.972e24  # Earth's mass
        positions = twobody_update(t, satellite_oe, t0, m_central)
        self.assertEqual(positions.shape, (2, 3))  # Two satellites, 3D positions

if __name__ == "__main__":
    unittest.main()


'''
#visualization
def plot_satellite_trajectory(oe_list, t_range, t0, m_central):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for oe in oe_list:
        positions = []
        for t in t_range:
            pos = satellite_position(t, oe, t0, m_central)
            positions.append(pos)
        positions = np.array(positions)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label=f"Satellite: {oe}")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.show()
'''


# Example usage:
# Orbital elements for satellites: [e, a, i, RAAN, argument of perigee, M0]
satellite_oe = [
    [0.01, 7000e3, np.radians(30), np.radians(40), np.radians(60), np.radians(0)],
    [0.02, 7100e3, np.radians(50), np.radians(80), np.radians(90), np.radians(10)]
]

# Central body (Earth) parameters
earth_mass = 5.972e24  # kg

'''
plot_satellite_trajectory(
    satellite_oe, 
    t_range=np.linspace(0, 3600, 100), 
    t0=0, 
    m_central=earth_mass
)
'''
