import numpy as np
import math

def gravitational_assist(mu_planet, v_in_spacecraft, v_planet, r_closest):
    """Calculate gravitational assist of a planet."""
    v_in_planet = v_in_spacecraft - v_planet
    v_in_magnitude = np.linalg.norm(v_in_planet)

    deflection_angle = 2 * math.asin(1 / (1 + (r_closest * v_in_magnitude**2) / mu_planet))
    print(deflection_angle)

    rotation_matrix = np.array([
        [math.cos(deflection_angle), -math.sin(deflection_angle), 0],
        [math.sin(deflection_angle), math.cos(deflection_angle), 0],
        [0, 0, 1]
    ])

    v_out_planet = np.dot(rotation_matrix, v_in_planet)
    v_out_sun = v_out_planet + v_planet

    return deflection_angle, v_out_sun


if __name__ == "__main__":
    # unittest.main()
    mu_planet = 3.986e14  # Example: Earth's gravitational parameter (m^3/s^2)
    v_in_spacecraft = np.array([12e3, 0, 0])  # 12 km/s
    v_planet = np.array([30e3, 0, 0])  # 30 km/s (approx Earth's orbital velocity)
    # r_closest = 7e6  # 7000 km (near Earth's surface)

    for r_closest in range(int(1e6), int(12e6), int(1e6)):
        deflection_angle, v_out_sun = gravitational_assist(mu_planet, v_in_spacecraft, v_planet, r_closest)
        print(f"Deflection angle: {deflection_angle:.3f} rad")
        print(f"Velocity after assist: {v_out_sun} m/s")
