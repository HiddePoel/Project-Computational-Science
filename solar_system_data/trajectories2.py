import numpy as np
import projectComputationalScience.solar_system_data.Bodies as Bodies

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
    start_planet = Bodies.Planet(start)
    end_planet = Bodies.Planet(end)

    rocket_speed_km_h = rocket_speed_km_s * 3600  # Convert to km/h
    rocket_speed = rocket_speed_km_h * 24  # Convert to km/day

    tolerance = tolerance_km

    # Outer boundary (limit to prevent runaway searches)
    outer_bound = distance_from_sun(end_planet.data[0]) * 1.1

    
    success = False
    iteration = 0
    best_vector = None
    closest_distance = np.inf

    days_look_ahead = 1

    while not success and iteration < max_iterations:
        vector = end_planet.data[days_look_ahead, :3]- start_planet.data[0, :3]
        vector = vector / np.linalg.norm(vector)

        current_closest_distance = np.inf

        position = start_planet.data[0, :3]

        for i in range(start_planet.data.shape[0]):
            position += vector * rocket_speed
            current_distance = distance(position, end_planet.data[i, :3])

            if current_distance < current_closest_distance:
                current_closest_distance = current_distance
                if current_closest_distance < closest_distance:
                    closest_distance = current_distance
                    best_vector = vector

            if current_distance < tolerance:
                success = True
                break

            if distance_from_sun(position) > outer_bound:
                print(f"Outer boundary reached at iteration {iteration}. Closest distance: {closest_distance} km.")
                print(f"days_look_ahead: {days_look_ahead}")
                break

        if current_closest_distance == closest_distance:
            days_look_ahead *= 2
        else:
            
            pass
            # days_look_ahead //= 2
            # days_look_ahead += (days_look_ahead //2)

        iteration += 1
    
    return best_vector



def distance_from_sun(position, axis=None):
    """Calculate distance from the Sun."""
    return np.linalg.norm(position, axis=axis)

def distance(position1, position2, axis=None):
    """Calculate distance between two points."""
    return np.linalg.norm(position1 - position2, axis=axis)

if __name__ == "__main__":

    calculate_launch_vector()
