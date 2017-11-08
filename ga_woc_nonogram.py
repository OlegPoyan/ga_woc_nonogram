from itertools import chain
from functools import reduce
from random import randint
import uuid
from PIL import Image, ImageDraw

EMPTY = 0
FILLED = 1

class Nonogram(object):
    """A board that represents n x n board for nonogram puzzle.

    Attributes:
        nonogram_id: unique identifier of the Nonogram
        row_numbers: array of tuples. Tuples represent each row's number
            of squares, and grouping
        column_numbers: arrya of tuples. Tuples represent each column's number
            of squares, and grouping
        grid: 2d array that represents n x n grid, with values 0 for EMPTY,
            1 for FILLED
        nonogram_size: dimensions of the grid
        fitness: fitness score #TODO
    """
    @staticmethod
    def create_rand_grid(grid_size):
        """Returns two dimensional array with squares in the grid filled in
        randomly."""

        return [[randint(0, 1) for x in range(0, grid_size)] for y in range(0, grid_size)]

    def __init__(self, nonogram_size):
        """Return board with dimensions of size nonogram_size. row_numbers,
            column_number are hardcoded for now."""
        # create random id
        self.nonogram_id = uuid.uuid4()
        self.row_numbers = [(2), (2), (2)]
        self.column_numbers = [(1, 1), (3), (1)]
        self.nonogram_size = nonogram_size
        self.grid = Nonogram.create_rand_grid(nonogram_size)
        #TODO
        self.fitness = 999
    def draw_nonogram(self):
        """ Create an PNG format image of grid"""
        image = Image.new("RGB", (self.nonogram_size * 50, self.nonogram_size * 50), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        for index, square in enumerate(reduce(lambda x, y: x+y, self.grid), 0):

            #print(square)
            x = index % self.nonogram_size
            y = index // self.nonogram_size
            coord = [(x * 50, y * 50), ((x + 1) * 50, (y + 1) * 50)]
            if square == EMPTY:
                draw.rectangle(coord, fill=(255, 255, 255))
            if square == FILLED:
                draw.rectangle(coord, fill=(0, 0, 0))
        return image

def create_population(board_size, population_size):
    """Returns a list of randomly filled Nonogram puzzle objects"""
    return [Nonogram(board_size) for x in range(0, population_size)]

# def ga_algorithm():

for board in create_population(10, 10):
    image = board.draw_nonogram()
    image.save("%s_%d.png" % ("nono", board.nonogram_id))
