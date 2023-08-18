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