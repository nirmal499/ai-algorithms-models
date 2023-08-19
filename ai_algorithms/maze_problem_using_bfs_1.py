from libs.pyamaze import maze
import libs.utils as utils

def run_bfs(m, visited_cells):
    q = []

    # Append the current node to the queue
    q.append(utils.MyCell((m.rows, m.cols)))

    # Keep searching while there are nodes in the stack
    while len(q) > 0:

        # Set the next node in the stack as the current node
        current_point_cell = q.pop(0)

        # If the current node hasn't already been exploited, search it
        if not utils.is_in_visited_cells(current_point_cell, visited_cells):
            visited_cells.append(current_point_cell)

            # Return the path to the current_point_cell if it is the goal
            if current_point_cell.my_tuple == (1,1):
                # return "Found"
                return current_point_cell
            else:

                # Get the neighbours
                # add available cells N, E, S, W to a list neighbors
                neighbors_cells = []
                for d in 'NESW': # here order matters
                    if m.maze_map[current_point_cell.my_tuple][d] == True:
                        
                        # available_point = tuple()

                        if d == 'E':
                            available_point = utils.MyCell((current_point_cell.my_tuple[0], current_point_cell.my_tuple[1] + 1))
                        if d == 'W':
                            available_point = utils.MyCell((current_point_cell.my_tuple[0], current_point_cell.my_tuple[1] - 1))
                        if d == 'N':
                            available_point = utils.MyCell((current_point_cell.my_tuple[0] - 1, current_point_cell.my_tuple[1]))
                        if d == 'S':
                            available_point = utils.MyCell((current_point_cell.my_tuple[0] + 1, current_point_cell.my_tuple[1]))

                        neighbors_cells.append(available_point)

                # Add the current node's neighbors to the stack
                for neighbor in neighbors_cells:

                    neighbor.set_parent(current_point_cell)
                    q.append(neighbor)

    return "No Path To Goal"

m = maze(6,6)
m.CreateMaze(loopPercent=30)

# print(m.maze_map)
# print(run_bfs(m))

outcome = run_bfs(m,[])

if type(outcome) is str:
    print(outcome)
else:
    bfs_path = utils.get_path(outcome)
    utils.my_print(bfs_path)

m.run()
