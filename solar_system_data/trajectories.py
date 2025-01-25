import numpy as np
import Planet

def calculate_launch_vector(
    start="Earth", end="Jupiter", rocket_speed_km_s=100, tolerance_km=500, max_iterations=100000
):
    """
    Simulates a rocket launch from one planet to another and finds the optimal launch direction.

    Args:
        start (str): Name of the starting planet.
        end (str): Name of the target planet.
        rocket_speed_km_s (float): Rocket speed in km/s.
        tolerance_km (float): Tolerance for closest approach in km.
        max_iterations (int): Maximum number of iterations for the optimization.

    Returns:
        np.ndarray: Optimal launch vector.
    """

    # Load planet data
    start_planet = Planet.Planet(start)
    end_planet = Planet.Planet(end)

    rocket_speed_km_h = rocket_speed_km_s * 3600  # Convert to km/h
    rocket_speed = rocket_speed_km_h * 24  # Convert to km/day

    tolerance = tolerance_km

    # Outer boundary (limit to prevent runaway searches)
    outer_bound = distance_from_sun(end_planet.data[0]) * 1.1

    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]  # Normalize vectors

    success = False
    iteration = 0
    best_vector = None
    closest_distance = np.inf

    while not success and iteration < max_iterations:
        distances = np.zeros((start_planet.data.shape[0], vectors.shape[0]))
        positions = np.repeat(start_planet.data[0, :3][np.newaxis, :], vectors.shape[0], axis=0)

        for i in range(start_planet.data.shape[0]):
            positions += vectors * rocket_speed
            current_distances = distance(positions, end_planet.data[i, :3], axis=1)
            distances[i] = current_distances

            if np.min(current_distances) < closest_distance:
                closest_distance = np.min(current_distances)
                best_vector = vectors[np.argmin(current_distances)]

            if np.any(current_distances < tolerance):
                success = True
                break

            if np.all(distance_from_sun(positions) > outer_bound):
                print(f"Outer boundary reached at iteration {iteration}. Closest distance: {closest_distance} km.")
                break

            if i % 10000 == 0:
                pass

        smallest_distances = np.min(distances, axis=0)
        best_indices = np.argsort(smallest_distances)[:3]
        vectors = vectors[best_indices]
        vectors = np.vstack([vectors, np.mean(vectors, axis=0)])
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]  # Normalize vectors

        iteration += 1

    if success:
        print(f"Launch vector found in {iteration} iterations with distance {closest_distance} km.")
    else:
        print(f"Failed to find a suitable launch vector within {max_iterations} iterations.")

    return best_vector

def distance_from_sun(position, axis=None):
    """Calculate distance from the Sun."""
    return np.linalg.norm(position, axis=axis)

def distance(position1, position2, axis=None):
    """Calculate distance between two points."""
    return np.linalg.norm(position1 - position2, axis=axis)

calculate_launch_vector()
