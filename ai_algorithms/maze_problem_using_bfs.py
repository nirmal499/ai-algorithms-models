from pyamaze import maze

class MyCell:
    def __init__(self, my_tuple):
        self.my_tuple = my_tuple
        self.parent = None

    def set_parent(self, parent_cell):
        self.parent = parent_cell

    def __str__(self):
        return str(self.my_tuple)

def is_in_visited_cells(neighbor, visited_cells):
    for cells in visited_cells:
        if neighbor.my_tuple == cells.my_tuple:
            return True

    return False

def my_print(cells_arr):
    print("[",end=" ")
    for cell in cells_arr:
        print(cell, end=", ")

    print("]")

# Utility to get a path as a list of points by traversing the parents of a node until the root is reached.
def get_path(point):
    path = []
    current_point = point
    while current_point.parent is not None:
        path.append(current_point)
        current_point = current_point.parent
    return path

def run_bfs(m, visited_cells):
    q = []
    # q.append((m.rows, m.cols))

    q.append(MyCell((m.rows, m.cols)))

    # marking current_point as visited
    visited_cells.append(MyCell((m.rows, m.cols)))

    # bfsPath = dict()

    while len(q) > 0:
        current_point_cell = q.pop(0)
        
        # add available cells N, E, S, W to a list neighbors
        neighbors_cells = []
        for d in 'NESW':
            if m.maze_map[current_point_cell.my_tuple][d] == True:
                
                # available_point = tuple()

                if d == 'E':
                    available_point = MyCell((current_point_cell.my_tuple[0], current_point_cell.my_tuple[1] + 1))
                if d == 'W':
                    available_point = MyCell((current_point_cell.my_tuple[0], current_point_cell.my_tuple[1] - 1))
                if d == 'N':
                    available_point = MyCell((current_point_cell.my_tuple[0] - 1, current_point_cell.my_tuple[1]))
                if d == 'S':
                    available_point = MyCell((current_point_cell.my_tuple[0] + 1, current_point_cell.my_tuple[1]))

                neighbors_cells.append(available_point)

        # my_print(neighbors_cells)

        for neighbor in neighbors_cells:
            if not is_in_visited_cells(neighbor, visited_cells):
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
    bfs_path = get_path(outcome)
    my_print(bfs_path)

m.run()
