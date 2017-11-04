from itertools import chain
import uuid
from PIL import Image, ImageDraw

class Nonogram(object):
    """A board that represents n x n board for nonogram puzzle.

    Attributes:
        nonogram_id: unique identifier of the Nonogram
        row_numbers: array of tuples. Tuples represent each row's number
            of squares, and grouping
        column_numbers: arrya of tuples. Tuples represent each column's number
            of squares, and grouping
        grid: 2d array that represents n x n grid, with values -1 for EMPTY,
            1 for FILLED
        nonogram_size: dimensions of the grid
        fitness: fitness score #TODO
    """
    @staticmethod
    def create_grid(grid_size):
        """Returns two dimensional array with squares in the grid filled in
        randomly."""
        return [[0 for x in range(0, grid_size)] for y in range(0, grid_size)]

    def __init__(self, nonogram_size):
        """Return board with dimensions of size nonogram_size. row_numbers,
            column_number are hardcoded for now."""
        # create random id
        self.nonogram_id = uuid.uuid4()
        self.row_numbers = [(2), (2), (2)]
        self.column_numbers = [(1, 1), (3), (1)]
        self.nonogram_size = nonogram_size
        self.grid = Nonogram.create_grid(nonogram_size)
        #TODO
        self.fitness = 999
    def draw_nonogram(self):
        """ Create an PNG format image of grid"""
        image = Image.new(1, (self.nonogram_size * 10, self.nonogram_size * 10), 1)
        draw = ImageDraw.Draw(image)

        for square in chain(self.grid):
            draw


board1 = Nonogram(3)
board1.draw_nonogram
