def population_metrics(boards):
    population = len(boards)
    best = boards[0].fitness
    worst = boards[population-1]
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
        median = (boards[population/2].fitness + boards[population/2+1].fitness) / 2
    else:
        median = boards[population/2].fitness
    np.std(fitnesses, ddof = 0)
