def population_metrics(boards, generation):
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
    standard_deviation = np.std(fitnesses, ddof = 1)
    print (standard_deviation)
    file = open('nonogram.log', 'a')
    file.write (generation + " " + best + " " + average + " " + worst + " " + median + " " + standard_deviation)
    file.close()
