from functools import reduce
from itertools import chain
from random import randint, random, choice
from math import floor
import uuid
import os
from PIL import Image, ImageDraw
import numpy as np
import time

EMPTY = 0
FILLED = 1
POPULATION_SIZE = 10
GEN_ITERATIONS = 10
REJECTION_RATE = 75
MUTATION_RATE = 0.1

SQUARE_PENALTY = 1
GROUP_PENALTY = 6
THRESHOLD = 0.7

# EXAMPLE PUZZLE: PANDA 35X47
# ROW_CONSTRAINTS = [
#     (1, 3), (1, 2, 2), (3, 3), (5, 1), (5, 2, 6), (11, 2, 4), (10, 7), (9, 3),
#     (9, 2), (9, 1), (8, 2), (8, 3, 1), (9, 4, 1), (9, 5, 2), (8, 2, 2, 1),
#     (8, 4, 1), (8, 1), (8, 1), (8, 1, 2), (9, 1, 2, 3), (9, 1, 2, 3),
#     (9, 1, 2, 3), (11, 1, 2, 2, 3), (12, 3, 2), (1, 6, 1, 2, 3), (1, 2, 4, 6),
#     (1, 2, 2, 5, 2), (1, 9), (2, 8, 1), (1, 10), (3, 8, 1), (4, 9), (4, 5, 1),
#     (5, 1, 4, 3), (8, 3, 5), (10, 1, 8), (11, 14), (12, 2, 15), (13, 16, 1),
#     (16, 12, 1,
#      2), (16, 11, 4), (14, 3, 14), (15, 2, 13), (13, 4,
#                                                  12), (13, 16), (17, ), (15, )
# ]
# COL_CONSTRAINTS = [
#     (5, ), (10, ), (2, 16), (10, 16), (12, 16), (16, 13), (18, 14), (20, 13),
#     (23, 12), (1, 23, 12), (24, 11), (2, 22, 10), (1, 2, 9, 3, 9),
#     (3, 6, 4, 2, 9), (7, 1, 2, 3), (2, 2, 3, 2, 4, 3), (1, 5, 4, 1, 3, 2, 2),
#     (2, 3, 1, 1, 2, 2, 4), (1, 5, 1, 3, 6), (1, 4, 3, 8), (1, 2, 2, 3, 5, 1),
#     (1, 4, 18), (2, 2, 9, 9), (2, 10, 9), (1, 9, 10), (2, 9, 11), (2, 8, 12),
#     (3, 3, 5, 12), (4, 2, 3, 1,
#                     14), (5, 2, 6, 11), (4, 5, 4,
#                                          4), (9, 10), (2, 2, 5), (7, ), (1, 3)
# ]

# EXAMPLE PUZZLE: FACE 10X10
# ROW_CONSTRAINTS = [(3, 3), (2, 4, 2), (1, 1), (1, 2, 2, 1), (1, 1, 1),
#                    (2, 2, 2), (1, 1), (1, 2, 1), (2, 2), (6, )]
# COL_CONSTRAINTS = [(5, ), (2, 4), (1, 1, 2), (2, 1, 1), (1, 2, 1, 1),
#                    (1, 1, 1, 1, 1), (2, 1, 1), (1, 2), (2, 4), (5, )]

# EXAMPLE PUZZLE: TANK 3X3
ROW_CONSTRAINTS = [(2, ), (2, ), (2, )]
COL_CONSTRAINTS = [(1, 1), (3, ), (1, )]
# EXAMPLE PUZZLE: TEST 3X5
# ROW_CONSTRAINTS = [(1, ), (1, 1), (2, ), (2, ), (1, 1)]
# COL_CONSTRAINTS = [(2, 1), (2, ), (4, )]


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
    def create_rand_grid(grid_width, grid_height):
        """Returns two dimensional array with squares in the grid filled in
        randomly."""

        return [[randint(0, 1) for x in range(0, grid_width)]
                for y in range(0, grid_height)]

    @staticmethod
    def create_grid(square_list, grid_width):
        """Returns 2d list with squares filled based on the binary string
        square list"""

        return [
            square_list[i:i + grid_width]
            for i in range(0, len(square_list), grid_width)
        ]

    @staticmethod
    def calc_fitness(self):
        """Returns the fitness for a particular grid"""

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
            # print(
            #     str(filled_count) + " - " + str(row_square_number) + " " +
            #     str(group_count) + " - " + str(len(self.row_numbers[index])))
            # TODO it will count len((0,)) to be one, needs to be 0
            score += SQUARE_PENALTY * abs(
                filled_count - row_square_number) + GROUP_PENALTY * abs(
                    group_count - len(self.row_numbers[index]))

        matrix = np.array(self.grid)
        # print (matrix)
        # print(matrix.T)
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
            # print(
            #     str(filled_count) + " - " + str(column_square_number) + " " +
            #     str(group_count) + " - " + str(
            #         len(self.column_numbers[index])))
            # TODO it will count len((0,)) to be one, needs to be 0
            score += SQUARE_PENALTY * abs(
                filled_count - column_square_number) + GROUP_PENALTY * abs(
                    group_count - len(self.column_numbers[index]))

        return score

    def __init__(self, square_list=None):
        """Return board with dimensions of size nonogram_size. row_numbers,
            column_number are hardcoded for now."""
        # create random id
        self.nonogram_id = uuid.uuid4()
        self.row_numbers = ROW_CONSTRAINTS
        self.column_numbers = COL_CONSTRAINTS
        self.grid_width = len(COL_CONSTRAINTS)
        self.grid_height = len(ROW_CONSTRAINTS)
        if square_list is None:
            self.grid = Nonogram.create_rand_grid(self.grid_width,
                                                  self.grid_height)
        else:
            self.grid = Nonogram.create_grid(square_list, self.grid_width)
        self.fitness = Nonogram.calc_fitness(self)
        # print("Creating board id: " + str(self.nonogram_id) + " fitness: " +
        #     str(self.fitness))

    def draw_nonogram(self):
        """ Create an PNG format image of grid"""
        image = Image.new("RGB", (self.grid_width * 10, self.grid_height * 10),
                          (255, 255, 255))
        draw = ImageDraw.Draw(image)

        for index, square in enumerate(
                reduce(lambda x, y: x + y, self.grid), 0):

            # print(square)
            x = index % self.grid_width
            y = index // self.grid_width
            coord = [(x * 10, y * 10), ((x + 1) * 10, (y + 1) * 10)]
            if square == EMPTY:
                draw.rectangle(coord, fill=(255, 255, 255))
            if square == FILLED:
                draw.rectangle(coord, fill=(0, 0, 0))
        return image


def population_metrics(boards, generation):
    population = len(boards)
    best = boards[0].fitness
    worst = boards[population - 1].fitness
    average = 0
    median = 0
    buffer = 0
    standard_deviation = 0
    fitnesses = []
    # find average
    for pop_size in range(0, population):
        buffer += boards[pop_size].fitness
        fitnesses.append(boards[pop_size].fitness)
    average = buffer / population
    # calculate median
    if (population % 2 == 0):
        median = (boards[int(population / 2)].fitness +
                  boards[int(population / 2 + 1)].fitness) / 2
    else:
        median = boards[int(population / 2)].fitness
    standard_deviation = np.std(fitnesses, ddof=1)
    # print(standard_deviation)
    file = open('nonogram.log', 'a')
    line_of_text = str(generation) + " " + str(best) + " " + str(
        average) + " " + str(worst) + " " + str(median) + " " + str(
            standard_deviation) + "\n"
    file.write(line_of_text)
    file.close()


def create_population(population_size):
    """Returns a list of randomly filled Nonogram puzzle objects"""
    return [Nonogram() for x in range(0, population_size)]


def reject_unfit(population, reject_percentage):
    """Returns a new list with the most fit individuals in the
    reject_percentage"""
    return population[0:floor((reject_percentage / 100) * len(population))]


def calc_total_fit(population):
    """ Returns total fitness score for the population."""
    total_fitness_score = 0
    for chromosome in population:
        total_fitness_score += chromosome.fitness
    return total_fitness_score


# def roulette_wheel_select(candidates):
#     """ Returns an individual from population and its index in a list.
#     The chance of being selected is proportional to the individual fitness."""
#     # TODO: Unfortuantely doesn't work as expected rn
#     print("ROULLETE WHEEL")
#     a = range(len(candidates))
#     lambd = sum(a) / len(candidates)
#     print("lambda = %f" % lambd)
#     index = expovariate(1 / lambd)
#     print(index)
#     return candidates.pop(int(index))
#     # roulette_arrow = uniform(0, fitness_range)
#     # current = 0
#     # for index, chromosome in enumerate(candidates):
#     #     if chromosome.fitness == 0:
#     #         # treat perfect solution as having fitness_score of 1
#     #         current += fitness_range - 1
#     #     else:
#     #         current += chromosome.fitness
#     #     if current > roulette_arrow:
#     #         return candidates.pop(index)


def mate(candidates):
    """ Returns 2 offsprings by mating 2 randomly choosen candidates """
    # print("\nStarting crossover")
    # print(candidates)

    chromosome1 = list(chain.from_iterable(choice(candidates).grid))
    chromosome2 = list(chain.from_iterable(choice(candidates).grid))

    offspring1, offspring2 = single_point_crossover(chromosome1, chromosome2)
    board1 = Nonogram(square_list=offspring1)
    board2 = Nonogram(square_list=offspring2)
    if board1.fitness < board2.fitness:
        return board1
    else:
        return board2


def single_point_crossover(chromosome1, chromosome2):
    """ Returns 2 chromosomes by randomly swapping genes """
    # print(chromosome1)
    # print(chromosome2)
    chromosome_len = len(chromosome1)
    crossover_begin = randint(0, chromosome_len)
    crossover_point = randint(crossover_begin, chromosome_len)
    crossover_end = randint(crossover_point, chromosome_len)
    # print(crossover_point)
    offspring1 = chromosome1[0:
                             crossover_begin] + chromosome2[crossover_begin:
                                                            crossover_point] + chromosome1[crossover_point:
                                                                                           crossover_end] + chromosome2[crossover_end:
                                                                                                                        chromosome_len]
    offspring2 = chromosome2[0:
                             crossover_begin] + chromosome1[crossover_begin:
                                                            crossover_point] + chromosome2[crossover_point:
                                                                                           crossover_end] + chromosome1[crossover_end:
                                                                                                                        chromosome_len]
    """offspring1 = chromosome1[0:crossover_point] + chromosome2[crossover_point:
                                                              chromosome_len]
    offspring2 = chromosome2[0:crossover_point] + chromosome1[crossover_point:
                                                              chromosome_len]"""
    # print("CROSSOVER RESULT")
    # print(offspring1)
    # print(offspring2)
    return offspring1, offspring2


def mutation(population):
    for board in population:
        if random() <= MUTATION_RATE:
            chromosome = np.ravel(board.grid)

    for board in population:
        if random() <= MUTATION_RATE:
            # print("Mutation triggered")
            chromosome = np.ravel(board.grid)
            i = 0
            while i < board.grid_height:
                j = randint(0, len(chromosome) - 1)
                if chromosome[j] == 1:
                    chromosome[j] = 0
                else:
                    chromosome[j] = 1
                i += 1
            chromosome = chromosome.tolist()
            chunks = [
                chromosome[x:x + board.grid_width]
                for x in range(0, board.grid_height)
            ]
            board.grid = chunks
            board.fitness = Nonogram.calc_fitness(board)


def ga_algorithm(population_size):
    """ga algorithm to find a solution for Nonogram puzzle"""

    # Start timer to measure performance
    t0 = time.time()
    population = create_population(population_size)
    population.sort(key=lambda individual: individual.fitness)
    draw_population(population, 'pics/gen_0/population/', 'nono')
    population_metrics(population, 0)

    for i in range(0, GEN_ITERATIONS):
        # print("Rejecting unfit candidates \n")
        population = reject_unfit(population, REJECTION_RATE)
        # path = 'pics/gen_' + str(i) + '/'
        # draw_population(population, path + 'fit_population/', 'fit_nono')

        # Create new chromosomes until reaching POPUlATION_SIZE
        next_gen = []
        while len(next_gen) < population_size - 1:
            next_gen.append(mate(population[:]))
        mutation(next_gen)
        # print("Create Adj Matrix\n")
        adj_matrix = wisdom_of_crowds(population)
        # print(adj_matrix)
        board = Nonogram()
        board.grid = wisdom_create_board(adj_matrix, THRESHOLD)
        board.fitness = Nonogram.calc_fitness(board)
        next_gen.append(board)
        next_gen.sort(key=lambda individual: individual.fitness)
        # print("Create new board and extend to population")
        # print("NEW POPULATION")
        population_metrics(next_gen, i + 1)
        # path = 'pics/gen_' + str(i + 1) + '/'
        # draw_population(next_gen, path + 'population/', 'nono')
        population = next_gen

    draw_population(next_gen, 'pics/last_gen/population/', 'nono')

    t1 = time.time()
    file = open('nonogram.log', 'a')
    file.write("Running time: " + str(t1 - t0) + "\n")
    file.write("POPULATION_SIZE " + str(POPULATION_SIZE) + "\n")
    file.write("GEN_ITERATIONS " + str(GEN_ITERATIONS) + "\n")
    file.write("REJECTION_RATE " + str(REJECTION_RATE) + "\n")
    file.write("MUTATION_RATE " + str(MUTATION_RATE) + "\n")
    file.write("\nSQUARE_PENALTY " + str(SQUARE_PENALTY) + "\n")
    file.write("GROUP_PENALTY " + str(GROUP_PENALTY) + "\n")
    file.write("WISDOM_TRHESHOLD " + str(THRESHOLD) + "\n")
    file.close()


def draw_population(population, path, filename):
    for index, board in enumerate(population):
        # Draw a picture of each individual in initial population
        image = board.draw_nonogram()
        if not os.path.exists(path):
            os.makedirs(path)
        image.save(path + filename + "_%d.png" % index)
        # print("Board #" + str(index) + " " + str(board.fitness))


def wisdom_of_crowds(population):
    grid_width = len(COL_CONSTRAINTS)
    grid_height = len(ROW_CONSTRAINTS)
    adj_matrix = [[0 for x in range(0, grid_width)]
                  for y in range(0, grid_height)]
    for pop_size in range(0, len(population)):
        for i in range(0, grid_height):
            for j in range(0, grid_width):
                if (population[pop_size].grid[i][j] == 1):
                    adj_matrix[i][j] += 1
    for i in range(0, grid_height):
        for j in range(0, grid_width):
            adj_matrix[i][j] /= len(population)
    return adj_matrix


def wisdom_create_board(adj_matrix, threshold):
    grid_width = len(COL_CONSTRAINTS)
    grid_height = len(ROW_CONSTRAINTS)
    board = [[0 for x in range(0, grid_width)] for y in range(0, grid_height)]
    for i in range(0, grid_height):
        for j in range(0, grid_width):
            if (adj_matrix[i][j] <= 1 - threshold):
                board[i][j] = 0
            elif (adj_matrix[i][j] > threshold):
                board[i][j] = 1
            else:
                board[i][j] = randint(0, 1)
    # print("wisdom ", board)
    return board


ga_algorithm(POPULATION_SIZE)
