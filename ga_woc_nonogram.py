from functools import reduce
from random import randint, choice, random
from math import floor
import uuid
import os
from PIL import Image, ImageDraw
import numpy as np

EMPTY = 0
FILLED = 1
POPULATION_SIZE = 6
BOARD_SIZE = 3


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

        return [[randint(0, 1) for x in range(0, grid_size)]
                for y in range(0, grid_size)]

    @staticmethod
    def calc_fitness(self):
        """Returns the fitness for a particular grid"""

        SQUARE_PENALTY = 1
        GROUP_PENALTY = 2

        score = 0

        for index, row in enumerate(self.grid):
            group_flag = False
            filled_count = 0
            group_count = 0
            row_square_number = 0

            for square in row:
                # Calculate number of groups and number of FILLED squares
                # present in the row
                if square == FILLED:
                    if not group_flag:
                        group_flag = True
                        group_count += 1
                    filled_count += 1
                else:
                    if group_flag:
                        group_flag = False
            for number in self.row_numbers[index]:
                row_square_number += number
            print(
                str(filled_count) + " - " + str(row_square_number) + " " +
                str(group_count) + " - " + str(len(self.row_numbers[index])))
            # TODO it will count len((0,)) to be one, needs to be 0
            score += SQUARE_PENALTY * abs(
                filled_count - row_square_number) + GROUP_PENALTY * abs(
                    group_count - len(self.row_numbers[index]))

        matrix = np.array(self.grid)
        for index, column in enumerate(matrix.T):
            group_flag = False
            filled_count = 0
            group_count = 0
            column_square_number = 0

            for square in column:
                # Calculate number of groups and number of FILLED squares
                # present in the column
                if square == FILLED:
                    if not group_flag:
                        group_flag = True
                        group_count += 1
                    filled_count += 1
                else:
                    if group_flag:
                        group_flag = False
            for number in self.column_numbers[index]:
                column_square_number += number
            print(
                str(filled_count) + " - " + str(column_square_number) + " " +
                str(group_count) + " - " + str(
                    len(self.column_numbers[index])))
            # TODO it will count len((0,)) to be one, needs to be 0
            score += SQUARE_PENALTY * abs(
                filled_count - column_square_number) + GROUP_PENALTY * abs(
                    group_count - len(self.column_numbers[index]))

        return score

    def __init__(self, nonogram_size):
        """Return board with dimensions of size nonogram_size. row_numbers,
            column_number are hardcoded for now."""
        # create random id
        self.nonogram_id = uuid.uuid4()
        print("Creating board id: " + str(self.nonogram_id))
        self.row_numbers = [(2, ), (2, ), (2, )]
        self.column_numbers = [(1, 1), (3, ), (1, )]
        self.nonogram_size = nonogram_size
        self.grid = Nonogram.create_rand_grid(nonogram_size)
        self.fitness = Nonogram.calc_fitness(self)
        self.probability = 0

    def draw_nonogram(self):
        """ Create an PNG format image of grid"""
        image = Image.new("RGB",
                          (self.nonogram_size * 50, self.nonogram_size * 50),
                          (255, 255, 255))
        draw = ImageDraw.Draw(image)

        for index, square in enumerate(
                reduce(lambda x, y: x + y, self.grid), 0):

            # print(square)
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


def reject_unfit(population, reject_percentage):
    """Returns a new list with the most fit individuals in the
    reject_percentage"""
    population.sort(key=lambda individual: individual.fitness)
    return population[0:floor((reject_percentage / 100) * len(population))]


def calc_total_fit(population):
    """ Returns total fitness score for the population."""
    total_fitness_score = 0
    for chromosome in population:
        total_fitness_score += chromosome.fitness
    return total_fitness_score


def calculate_fit_ratio(population):
    """ Assigns probability of being selected for mating for each individual
    Nonogram object in the population """
    total_fitness_score = calc_total_fit(population)
    print(total_fitness_score)
    for chromosome in population:
        chromosome.probability = chromosome.fitness / total_fitness_score
        print(chromosome.probability)


def crossover(population, board_size):
    # TODO: implement picking chromosome based on
    # Nonogram.probability attribute
    father = choice(population)
    child = father
    mother = choice(population)
    crossover_index = randint(0, len(population))
    father1D = np.ravel(father.grid)
    mother1D = np.ravel(mother.grid)
    begin_father = father1D[:crossover_index]
    father = begin_father.tolist()
    end_mother = mother1D[crossover_index:]
    mother = end_mother.tolist()
    baby = father + mother
    chunks = [baby[x:x + board_size] for x in range(0, len(baby), board_size)]
    child.grid = chunks
    return child


def mutation(population, population_size, board_size):
    mutation_rate = 0.01
    for index, board in enumerate(population):
        mutant = population[index]
        if random() <= mutation_rate:
            mutant1D = np.ravel(mutant.grid)
            j = random.randint(0, len(mutant1D) - 1)
            if mutant1D[j] == 1:
                mutant1D[j] = 0
            else:
                mutant1D[j] = 1
            mutant1D = mutant1D.tolist()
            chunks = [
                mutant1D[x:x + board_size]
                for x in range(0, len(mutant1D), board_size)
            ]
            mutant.grid = chunks


def ga_algorithm(board_size, population_size):
    """ga algorithm to find a solution for Nonogram puzzle"""
    population = create_population(board_size, population_size)
    draw_population(population, 'initial_population/', 'nono_init')

    print("Rejecting unfit candidates \n")
    population = reject_unfit(population, 50)
    print("New Population size: " + str(len(population)))
    calculate_fit_ratio(population)

    draw_population(population, 'fit_population/', 'fit_nono')

    # Create new chromosomes until reaching POPUlATION_SIZE
    new_population = []
    for i in range(population_size):
        new_population.append(
            crossover(population, board_size))

    mutation(new_population, population_size, board_size)


def draw_population(population, path, filename):
    for index, board in enumerate(population):
        # Draw a picture of each individual in initial population
        image = board.draw_nonogram()
        if not os.path.exists(path):
            os.mkdir(path)
        image.save(path + filename + "_%d.png" % index)
        print("Board #" + str(index) + " " + str(board.fitness))


ga_algorithm(BOARD_SIZE, POPULATION_SIZE)
