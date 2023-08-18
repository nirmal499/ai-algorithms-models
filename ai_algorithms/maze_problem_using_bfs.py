from libs.pyamaze import maze
import libs.utils as utils

def run_bfs(m, visited_cells):
    q = []
    # q.append((m.rows, m.cols))

    q.append(utils.MyCell((m.rows, m.cols)))

    # marking current_point as visited
    visited_cells.append(utils.MyCell((m.rows, m.cols)))

    # bfsPath = dict()

    while len(q) > 0:
        current_point_cell = q.pop(0)
        
        # add available cells N, E, S, W to a list neighbors
        neighbors_cells = []
        for d in 'NESW':
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

        # my_print(neighbors_cells)

        for neighbor in neighbors_cells:
            if not utils.is_in_visited_cells(neighbor, visited_cells):
                # current_point_cell = neighbor

                neighbor.set_parent(current_point_cell)

                # mark neighbor as visited
                visited_cells.append(neighbor)
                q.append(neighbor)

                # print(neighbor)
                
                # Check if the value at neighbor is the goal
                if neighbor.my_tuple == (1,1):
                    # return "Found"
                    return neighbor

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
