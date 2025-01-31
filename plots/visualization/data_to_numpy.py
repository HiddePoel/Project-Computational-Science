import numpy as np
import re
import json
import os


# Function to parse Horizons data into a NumPy array (positions and velocities only)
def parse_horizons_to_numpy(json_data):
    # Extract the result string
    result_string = json_data["result"]

    # Locate the start and end of the data block
    start_marker = "$$SOE"
    end_marker = "$$EOE"
    start_index = result_string.find(start_marker) + len(start_marker)
    end_index = result_string.find(end_marker)
    data_block = result_string[start_index:end_index].strip()

    # Regular expression to parse each data block
    pattern = (
        r"(?P<time>\d+\.\d+)\s+="  # Match time (Julian Date)
        r".*?X\s*=\s*(?P<x>[-+]?[0-9]*\.?[0-9]+E[-+]?[0-9]+)"  # Match X
        r"\s*Y\s*=\s*(?P<y>[-+]?[0-9]*\.?[0-9]+E[-+]?[0-9]+)"  # Match Y
        r"\s*Z\s*=\s*(?P<z>[-+]?[0-9]*\.?[0-9]+E[-+]?[0-9]+)"  # Match Z
        r".*?VX\s*=\s*(?P<vx>[-+]?[0-9]*\.?[0-9]+E[-+]?[0-9]+)"  # Match VX
        r"\s*VY\s*=\s*(?P<vy>[-+]?[0-9]*\.?[0-9]+E[-+]?[0-9]+)"  # Match VY
        r"\s*VZ\s*=\s*(?P<vz>[-+]?[0-9]*\.?[0-9]+E[-+]?[0-9]+)"  # Match VZ"
    )

    # Parse the data block
    matches = re.finditer(pattern, data_block, re.DOTALL)

    # Create a list for positions and velocities
    parsed_data = []
    for match in matches:
        x = float(match.group("x"))
        y = float(match.group("y"))
        z = float(match.group("z"))
        vx = float(match.group("vx"))
        vy = float(match.group("vy"))
        vz = float(match.group("vz"))
        parsed_data.append([x, y, z, vx, vy, vz])

    # Convert the list to a NumPy array with float64 type
    np_array = np.array(parsed_data, dtype=np.float64)

    return np_array


# Function to load JSON and parse into NumPy, then save
def process_and_save_numpy(json_file_path):
    # Load JSON data
    with open(json_file_path, "r") as file:
        json_data = json.load(file)

    # Parse into NumPy array
    np_array = parse_horizons_to_numpy(json_data)

    # Save NumPy array in the same location
    output_path = os.path.splitext(json_file_path)[0] + ".npy"
    np.save(output_path, np_array)
    print(f"NumPy array saved to {output_path}.")


# Example usage
# Directory containing the JSON files
planet_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planet_data")

# Process all JSON files in the directory
for file_name in os.listdir(planet_data_dir):
    if file_name.endswith(".json"):
        json_file_path = os.path.join(planet_data_dir, file_name)
        process_and_save_numpy(json_file_path)
