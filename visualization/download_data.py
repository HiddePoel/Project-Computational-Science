import requests
import os

# List of planet IDs
planet_ids = [199, 299, 399, 499, 599, 699, 799, 899]

# API base URL
base_url = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Query parameters template
query_template = {
    "format": "json",
    "COMMAND": None,  # To be set for each planet
    "OBJ_DATA": "YES",
    "MAKE_EPHEM": "YES",
    "EPHEM_TYPE": "VECTORS",
    "CENTER": "500@10",  # Sun-centered coordinate system
    "START_TIME": "1900-01-01",
    "STOP_TIME": "2025-12-31",
    "STEP_SIZE": "1d",  # Daily steps
}

# Output directory (current script's location)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planet_data")
os.makedirs(output_dir, exist_ok=True)


# Function to download data for a single planet
def download_planet_data(planet_id):
    query = query_template.copy()
    query["COMMAND"] = str(planet_id)
    response = requests.get(base_url, params=query)

    if response.status_code == 200:
        file_name = os.path.join(output_dir, f"planet_{planet_id}.json")
        with open(file_name, "w") as file:
            file.write(response.text)
        print(f"Data for planet {planet_id} saved to {file_name}.")
    else:
        print(f"Failed to fetch data for planet {planet_id}: {response.status_code}")


# Download data for each planet
for planet_id in planet_ids:
    download_planet_data(planet_id)

print("All planet data downloaded.")
