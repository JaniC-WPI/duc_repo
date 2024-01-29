#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import cv2

# Define the maze
# maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

maze = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# Convert the maze to a numpy array for visualization
maze_array = np.array(maze)

# Scale factor for each cell in the maze for visualization
scale_factor = 50

# Create an empty image for the maze visualization
maze_visualization = np.zeros((len(maze) * scale_factor, len(maze[0]) * scale_factor, 3), dtype=np.uint8)

# Fill the image with colors: white for passable cells, black for walls
for i in range(len(maze)):
    for j in range(len(maze[i])):
        color = (255, 255, 255) if maze[i][j] == 0 else (0, 0, 0)
        cv2.rectangle(maze_visualization, (j * scale_factor, i * scale_factor),
                      ((j + 1) * scale_factor - 1, (i + 1) * scale_factor - 1),
                      color, -1)

# Draw grid lines for clarity
for i in range(0, len(maze) * scale_factor, scale_factor):
    cv2.line(maze_visualization, (0, i), (maze_visualization.shape[1], i), (0, 0, 0), 1)
for j in range(0, len(maze[0]) * scale_factor, scale_factor):
    cv2.line(maze_visualization, (j, 0), (j, maze_visualization.shape[0]), (0, 0, 0), 1)

    # Define the planned path
# path = [(2, 0), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (8, 4), (8, 5), (8, 6), (7, 6)]
# path = [(2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (8, 4), (8, 5), (8, 6), (7, 6)]
# path = [(2, 0), (3, 1), (4, 2), (5, 3), (6, 3), (7, 3), (8, 4), (7, 5), (7, 6)]
# path = [(2, 0), (3, 1), (4, 2), (5, 3), (6, 3), (7, 3), (8, 4), (7, 5), (7, 6)]
path = [(2, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 6), (6, 6), (7, 6)]

# Draw the path on the maze visualization
for position in path:
    i, j = position
    cv2.circle(maze_visualization, 
               (j * scale_factor + scale_factor // 2, i * scale_factor + scale_factor // 2), 
               scale_factor // 4, 
               (0, 255, 0), 
               -1)

# Display the maze with grid
cv2.imshow('Maze with Grid', maze_visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a plot
# plt.figure(figsize=(5, 5))
# plt.imshow(maze_array, cmap='Greys', interpolation='none')

# # Add grid lines for clarity
# plt.grid(which='major', color='black', linestyle='-', linewidth=1)

# # Label the axes
# plt.xticks(range(len(maze[0])))
# plt.yticks(range(len(maze)))

# # Title
# plt.title("Maze Visualization")

# plt.show()



