<<<<<<< HEAD
from functools import reduce
from itertools import chain
from random import randint, random, uniform, expovariate, choice
from math import floor
import uuid
import os
from PIL import Image, ImageDraw
import numpy as np

EMPTY = 0
FILLED = 1
POPULATION_SIZE = 10
BOARD_SIZE = 3
GEN_ITERATIONS = 10

SQUARE_PENALTY = 1
GROUP_PENALTY = 6


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
    def create_grid(square_list, grid_size):
        """Returns 2d list with squares filled based on the binary string
        square list"""

        return [
            square_list[i:i + grid_size]
            for i in range(0, len(square_list), grid_size)
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

    def __init__(self, nonogram_size, square_list=None):
        """Return board with dimensions of size nonogram_size. row_numbers,
            column_number are hardcoded for now."""
        # create random id
        self.nonogram_id = uuid.uuid4()
        self.row_numbers = [(2, ), (2, ), (2, )]
        self.column_numbers = [(1, 1), (3, ), (1, )]
        self.nonogram_size = nonogram_size
        if square_list is None:
            self.grid = Nonogram.create_rand_grid(nonogram_size)
        else:
            self.grid = Nonogram.create_grid(square_list, nonogram_size)
        self.fitness = Nonogram.calc_fitness(self)
        print("Creating board id: " + str(self.nonogram_id) + " fitness: " +
              str(self.fitness))

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


def mate(candidates, board_size):
    """ Returns 2 offsprings by mating 2 randomly choosen candidates """
    print("\nStarting crossover")
    # print(candidates)

    chromosome1 = list(chain.from_iterable(choice(candidates).grid))
    chromosome2 = list(chain.from_iterable(choice(candidates).grid))

    offspring1, offspring2 = single_point_crossover(chromosome1, chromosome2)
    return Nonogram(
        board_size, square_list=offspring1), Nonogram(
            board_size, square_list=offspring2)


def single_point_crossover(chromosome1, chromosome2):
    """ Returns 2 chromosomes by randomly swapping genes """
    print(chromosome1)
    print(chromosome2)
    chromosome_len = len(chromosome1)
    crossover_point = randint(0, chromosome_len)
    print(crossover_point)
    offspring1 = chromosome1[0:crossover_point] + chromosome2[crossover_point:
                                                              chromosome_len]
    offspring2 = chromosome2[0:crossover_point] + chromosome1[crossover_point:
                                                              chromosome_len]
    print("CROSSOVER RESULT")
    print(offspring1)
    print(offspring2)
    return offspring1, offspring2


def mutation(population, population_size, board_size):
    mutation_rate = .01
    for index, board in enumerate(population):
        mutant = population[index]
        if random() <= mutation_rate:
            mutant1D = np.ravel(mutant.grid)
            j = randint(0, len(mutant1D) - 1)
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
            mutant.fitness = Nonogram.calc_fitness(mutant)


def ga_algorithm(board_size, population_size):
    """ga algorithm to find a solution for Nonogram puzzle"""
    population = create_population(board_size, population_size)
    draw_population(population, 'pics/gen_0/population/', 'nono')

    for i in range(0, GEN_ITERATIONS):
        print("gen_" + str(i))
        print("Rejecting unfit candidates \n")
        population.sort(key=lambda individual: individual.fitness)
        population = reject_unfit(population, 80)
        path = 'pics/gen_' + str(i) + '/'
        draw_population(population, path + 'fit_population/', 'fit_nono')

        # Create new chromosomes until reaching POPUlATION_SIZE
        next_gen = []
        while len(next_gen) < population_size:
            next_gen.extend(mate(population[:], board_size))

        mutation(next_gen, population_size, board_size)
        print("NEW POPULATION")
        path = 'pics/gen_' + str(i + 1) + '/'
        draw_population(next_gen, path + 'population/', 'nono')
        population = next_gen


def draw_population(population, path, filename):
    for index, board in enumerate(population):
        # Draw a picture of each individual in initial population
        image = board.draw_nonogram()
        if not os.path.exists(path):
            os.makedirs(path)
        image.save(path + filename + "_%d.png" % index)
        print("Board #" + str(index) + " " + str(board.fitness))


ga_algorithm(BOARD_SIZE, POPULATION_SIZE)
=======
from functools import reduce
from itertools import chain
from random import randint, random, uniform, expovariate, choice
from math import floor
import uuid
import os
from PIL import Image, ImageDraw
import numpy as np

EMPTY = 0
FILLED = 1
POPULATION_SIZE = 20
BOARD_SIZE = 3
GEN_ITERATIONS = 10
REJECTION_RATE = 80

SQUARE_PENALTY = 1
GROUP_PENALTY = 6


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
    def create_grid(square_list, grid_size):
        """Returns 2d list with squares filled based on the binary string
        square list"""

        return [
            square_list[i:i + grid_size]
            for i in range(0, len(square_list), grid_size)
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

    def __init__(self, nonogram_size, square_list=None):
        """Return board with dimensions of size nonogram_size. row_numbers,
            column_number are hardcoded for now."""
        # create random id
        self.nonogram_id = uuid.uuid4()
        print("Creating board id: " + str(self.nonogram_id))
        self.row_numbers = [(2, ), (2, ), (2, )]
        self.column_numbers = [(1, 1), (3, ), (1, )]
        self.nonogram_size = nonogram_size
        if square_list is None:
            self.grid = Nonogram.create_rand_grid(nonogram_size)
        else:
            self.grid = Nonogram.create_grid(square_list, nonogram_size)
        self.fitness = Nonogram.calc_fitness(self)

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


def population_metrics(boards, generation):
    population = len(boards)
    best = boards[0].fitness
    worst = boards[population-1].fitness
    average = 0
    median = 0
    buffer = 0
    standard_deviation = 0
    fitnesses = []
    # find average
    for pop_size in range (0, population):
        buffer += boards[pop_size].fitness
        fitnesses.append(boards[pop_size].fitness)
    average = buffer / population
    # calculate median
    if (population%2 == 0):
        median = (boards[int(population/2)].fitness + boards[int(population/2+1)].fitness) / 2
    else:
        median = boards[int(population/2)].fitness
    standard_deviation = np.std(fitnesses, ddof = 1)
    print (standard_deviation)
    file = open('nonogram.log', 'a')
    line_of_text = str(generation) + " " + str(best) + " " + str(average) + " " + str(worst) + " " + str(median) + " " + str(standard_deviation) + "\n"
    file.write (line_of_text)
    file.close()

def create_population(board_size, population_size):
    """Returns a list of randomly filled Nonogram puzzle objects"""
    return [Nonogram(board_size) for x in range(0, population_size)]


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


def mate(candidates, board_size):
    """ Returns 2 offsprings by mating 2 randomly choosen candidates """
    print("\nStarting crossover")
    # print(candidates)

    chromosome1 = list(
        chain.from_iterable(choice(candidates).grid))
    chromosome2 = list(
        chain.from_iterable(choice(candidates).grid))

    offspring1, offspring2 = single_point_crossover(chromosome1, chromosome2)
    return Nonogram(
        board_size, square_list=offspring1), Nonogram(
            board_size, square_list=offspring2)


def single_point_crossover(chromosome1, chromosome2):
    """ Returns 2 chromosomes by randomly swapping genes """
    print(chromosome1)
    print(chromosome2)
    chromosome_len = len(chromosome1)
    crossover_point = randint(0, chromosome_len)
    print(crossover_point)
    offspring1 = chromosome1[0:crossover_point] + chromosome2[crossover_point:
                                                              chromosome_len]
    offspring2 = chromosome2[0:crossover_point] + chromosome1[crossover_point:
                                                              chromosome_len]
    print("CROSSOVER RESULT")
    print(offspring1)
    print(offspring2)
    return offspring1, offspring2


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
    draw_population(population, 'pics/gen_0/population/', 'nono')

    for i in range(0, GEN_ITERATIONS):
        print("Rejecting unfit candidates \n")
        population.sort(key=lambda individual: individual.fitness)
        population_metrics(population, i)
        population = reject_unfit(population, REJECTION_RATE)
        path = 'pics/gen_' + str(i) + '/'
        draw_population(population, path + 'fit_population/', 'fit_nono')

        # Create new chromosomes until reaching POPUlATION_SIZE
        next_gen = []
        while len(next_gen) < population_size:
            next_gen.extend(mate(population[:], board_size))
        print("NEW POPULATION")
        path = 'pics/gen_' + str(i + 1) + '/'
        draw_population(next_gen, path + 'population/', 'nono')
        population = next_gen

        # mutation(next_gen, population_size, board_size)


def draw_population(population, path, filename):
    for index, board in enumerate(population):
        # Draw a picture of each individual in initial population
        image = board.draw_nonogram()
        if not os.path.exists(path):
            os.makedirs(path)
        image.save(path + filename + "_%d.png" % index)
        print("Board #" + str(index) + " " + str(board.fitness))


ga_algorithm(BOARD_SIZE, POPULATION_SIZE)
>>>>>>> b7a3b75d434c1aba38d2281af9a6759c6effc460
