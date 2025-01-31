import pygame
import numpy as np
import Bodies as Bodies

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 1000
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Solar System Visualization")

planet_names = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
planet_names = planet_names[:5]  # Limit to inner planets for better visibility
planets = {name: Bodies.Body(name) for name in planet_names}

# Calculate maximum distance from the Sun to any planet
max_distance = 0
for planet in planets.values():
    distances = np.sqrt(
        planet.data[:, 0] ** 2 + planet.data[:, 1] ** 2
    )  # Calculate distance from the Sun
    max_distance = max(max_distance, np.max(distances))

# Scaling factor to fit all planets within the window
scale = (WIDTH // 2) / max_distance  # Use half-width to account for both sides

# Normalize positions to center the Sun and scale planets
normalized_positions = []
for planet in planets.values():
    normalized_pos = planet.data * scale  # Scale positions
    normalized_pos[:, 0] += WIDTH // 2  # Center Sun horizontally
    normalized_pos[:, 1] += HEIGHT // 2  # Center Sun vertically
    normalized_positions.append(normalized_pos)

for i, planet in enumerate(planets.values()):
    planet.data = normalized_positions[i]

# Main loop
running = True
clock = pygame.time.Clock()

i = 0
max_index = planets["Earth"].data.shape[0]


while running:
    clock.tick(60)
    screen.fill(BLACK)

    # drawing the sun
    pygame.draw.circle(screen, YELLOW, (WIDTH // 2, HEIGHT // 2), 10)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw each planet's position at time step `i`
    for planet in planets.values():
        x, y = planet.data[i, 0:2]
        pygame.draw.circle(screen, planet.color, (int(x), int(y)), 5)

    i += 1
    if i >= max_index:
        i = 0

    pygame.display.update()

pygame.quit()
