#!/usr/bin/env python3

class Node:
    # Initialize a node with position, parent, g, h, and f scores
    def __init__(self, position=None, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost estimate to goal
        self.f = 0  # Total cost (g + h)

def heuristic(a, b):
    # Euclidean distance heuristic
    # return ((a[0] - b[0])**2) + ((a[1] - b[1])**2)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze, start, end):
    # Create start and end node
    start_node = Node(start, None)
    end_node = Node(end, None)

    # Initialize open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until the end is reached
    while len(open_list) > 0:
        # Get the current node (node with the lowest f score)
        # current_node = open_list[0]
        current_node = min(open_list, key=lambda node: node.f)
        open_list.remove(current_node)
        closed_list.append(current_node)

        # Check if we reached the goal
        if current_node.position == end_node.position:
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path, 

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Adjacent squares
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within the range of the maze
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(node_position, current_node)
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if child in closed_list:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = heuristic(child.position, end_node.position)
            child.f = child.g + child.h

            # Child is already in the open list
            if any([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]):
                continue

            # Add the child to the open list
            open_list.append(child)

    return None  # No path found

# Example usage
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



start = (2, 3)
end = (7, 6)

path = astar(maze, start, end)
print(path)
