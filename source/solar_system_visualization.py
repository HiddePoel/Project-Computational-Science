import pygame
import numpy as np
import os

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 1000
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Solar System Visualization")

celestial_body_names = [
    "sun",
    "mercury",
    "venus",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
]

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == "nt":
    filenames = [
        script_dir + "\\solar_system_data\\" + body_name + "_coordinates.npy"
        for body_name in celestial_body_names
    ]
else:
    filenames = [
        script_dir + "/solar_system_data/" + body_name + "_coordinates.npy"
        for body_name in celestial_body_names
    ]

min_x, min_y = np.inf, np.inf
max_x, max_y = -np.inf, -np.inf

# Load planet_positions from numpy files
planet_positions = []
for filename in filenames:
    pos = np.load(filename)
    planet_positions.append(pos)

    current_min_x, current_min_y = np.min(pos[:, 0:2], axis=0)
    current_max_x, current_max_y = np.max(pos[:, 0:2], axis=0)

    min_x = min(current_min_x, min_x)
    min_y = min(current_min_y, min_y)
    max_x = max(current_max_x, max_x)
    max_y = max(current_max_y, max_y)

# Calculate the scaling factor
scale_x = WIDTH / (max_x - min_x)
scale_y = HEIGHT / (max_y - min_y)
scale = min(scale_x, scale_y)

# Normalize planet_positions to fit within the window
normalized_positions = []
for pos in planet_positions:
    normalized_pos = (pos - [min_x, min_y, 0]) * scale
    normalized_positions.append(normalized_pos)
planet_positions = normalized_positions

# Main loop
running = True
clock = pygame.time.Clock()

i = 0
max_index = len(planet_positions[0])

while running:
    # print(i)
    clock.tick(60)
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for position in planet_positions:
        x, y = position[i, 0:2]
        pygame.draw.circle(screen, WHITE, (int(x), int(y)), 5)

    i += 100
    if i >= max_index:
        i = 0

    pygame.display.update()

pygame.quit()
