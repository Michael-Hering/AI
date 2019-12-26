import numpy as np
from utils import *
# import matplotlib.pyplot as plt
# import matplotlib.ticker as plticker
import random

def randomGenome(length):
    """
    :param length:
    :return: string, random binary digit
    """
    s = ""
    for i in range(length):
      s = s + str(np.random.randint(2))
    return s


def makePopulation(size, length):
    """
    :param size - of population:
    :param length - of genome
    :return: list of length size containing genomes of length length
    """
    pop = []
    for i in range(size):
      pop.append(randomGenome(length))
    return pop

def fitness(genome):
    """
    :param genome: 
    :return: the fitness value of a genome
    """
    fit = 0
    for i in range(len(genome)):
      if genome[i:i+1] == "1":
        fit += 1
    return fit

def evaluateFitness(population):
    """
    :param population: 
    :return: a pair of values: the average fitness of the population as a whole and the fitness of the best individual in the population.
    """
    bestFitness = 0
    n = len(population)
    totalFitness = 0
    for i in range(len(population)):
      thisFitness = fitness(population[i])
      if (thisFitness > bestFitness):
        bestFitness = thisFitness
      totalFitness += fitness(population[i])
    avg = totalFitness / n
    return (avg, bestFitness)




def crossover(genome1, genome2):
    """
    :param genome1:
    :param genome2:
    :return: two new genomes produced by crossing over the given genomes at a random crossover point.
    """
    k = np.random.randint(1, len(genome1)-1) # Choose a point that is not the start or end
    c1 = genome2[:k] + genome1[k:]
    c2 = genome1[:k] + genome2[k:]
    return (c1, c2)


def mutate(genome, mutationRate):
    """
    :param genome:
    :param mutationRate:
    :return: a new mutated version of the given genome.
    """
    mutated = ""
    for k in range(len(genome)):
      mutate = np.random.uniform(0,1) <= mutationRate
      if mutate:
        if genome[k] == "0":
          mutated = mutated + "1"
        else:
          mutated = mutated + "0"
      else:
        mutated = mutated + genome[k]
    return mutated


def selectPair(population):
    """

    :param population:
    :return: two genomes from the given population using fitness-proportionate selection. This function should use weightedChoice, which we wrote in class, as a helper function.
    """
    avgFitness, maxFitness = evaluateFitness(population)
    weights = []
    for i in range(len(population)):
      weights.append(fitness(population[i]) / maxFitness)
    return (weightedChoice(population, weights), weightedChoice(population, weights))

def runGA(populationSize, crossoverRate, mutationRate, logFile=""):
    """

    :param populationSize: :param crossoverRate: :param mutationRate: :param logFile: :return: xt file in which to
    store the data generated by the GA, for plotting purposes. When the GA terminates, this function should return
    the generation at which the string of all ones was found.is the main GA program, which takes the population size,
    crossover rate (pc), and mutation rate (pm) as parameters. The optional logFile parameter is a string specifying
    the name of a te
    """
    outFile = open(logFile, 'w+')
    genomeLength = 20
    population = makePopulation(populationSize, genomeLength)
    print("Population size: {}".format(populationSize))
    print("Genome length: {}".format(genomeLength))
    converged = False
    max_iter = 1000
    current_iter = 0
    while (not converged and current_iter < max_iter):
      avgFitness, maxFitness = evaluateFitness(population)
      outFile.write("{} {} {} \n".format(current_iter, avgFitness, maxFitness))
      print("Generation  {}: average fitness {}, best fitness {}".format(current_iter, avgFitness, maxFitness))
      if (maxFitness == genomeLength):
        converged = True
      else:
        newPopulation = []
        for _ in range(populationSize):
          parent1, parent2 = selectPair(population)
          perform_crossover = np.random.uniform(0,1) <= crossoverRate
          if perform_crossover:
            child1, child2 = crossover(parent1, parent2)
          else:
            child1, child2 = parent1, parent2
          child1 = mutate(child1, mutationRate)
          child2 = mutate(child2, mutationRate)
          newPopulation.append(child1)
          newPopulation.append(child2)
        population = newPopulation
        current_iter += 1
    return current_iter



# def plot_random_5(path):

#   random_5 = random.sample(list(np.arange(0,50)), 5)
#   x_dict={}
#   y_dict={}
#   k = 0

#   for run in random_5:
#     x_dict[run] = []
#     y_dict[run] = []
#     inFile = open(path+"run{}.txt".format(run))
#     line = inFile.readline()
#     while line:
#       gen, avg_fitness, max_fitness = line.split()
#       x_dict[run].append(float(gen))
#       y_dict[run].append(float(avg_fitness))
#       line = inFile.readline()
#     k+=1

#   fig, ax = plt.subplots()
#   for run in random_5:
#     ax.plot(x_dict[run], y_dict[run], label = "run {}".format(run))
#   ax.set_xlabel("Generation")
#   ax.set_ylabel("Average fitness")
#   ax.set_title("Average Fitness vs Generation")
#   plt.legend()
#   plt.show()


if __name__ == '__main__':
  None
    ### 1. ###

    # gen_list_7 = []
    # for i in range(50):
    #   filename = "results/part1/crossover_7/run{}.txt".format(i)
    #   gen = runGA(100, 0.7, 0.001, filename)
    #   gen_list_7.append(gen)

    # print("Report for 50 runs with 0.7 crossover rate: Minimum generation: {}, Average generation: {}, Maximum Generation: {}".format(min(gen_list_7), Average(gen_list_7), max(gen_list_7)))

    ### 2. ####

    # plot_random_5("results/part1/crossover_7/")

    ### 3. ###

    # gen_list_0 = []
    # for i in range(50):
    #   filename = "results/part1/crossover_0/run{}.txt".format(i)
    #   gen = runGA(100, 0, 0.001, filename)
    #   gen_list_0.append(gen)
    
    # print("Report for 50 runs with 0 crossover rate: Minimum generation: {}, Average generation: {}, Maximum Generation: {}".format(min(gen_list_0), Average(gen_list_0), max(gen_list_0)))

    # plot_random_5("results/part1/crossover_0/")

    ### 4. ###

    # mutation_rates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.02]
    # crossover_rates = [0, 0.125, 0.25, 0.5, 0.625, 0.75, 0.875, 1]
    # population_sizes = [10, 30, 50, 70, 90, 110, 130, 150]

    # baseline_mutation_rate = 0.001
    # baseline_crossover_rate = 0.7
    # baseline_population_size = 100

    # gen_list_mutation = []
    # for mutation_rate in mutation_rates:
    #   total_iterations = 0
    #   for i in range(20):
    #     filename = "results/part1/4/mutation_rate_{}_run{}.txt".format(mutation_rate, i)
    #     gen = runGA(baseline_population_size, baseline_crossover_rate, mutation_rate, filename)
    #     total_iterations += gen
    #   gen_list_mutation.append(total_iterations / 50) # Store the average  iterations
    
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(mutation_rates, gen_list_mutation)
    # ax.set_xlabel("Mutation rate")
    # ax.set_ylabel("Average number of generations to reach goal")
    # ax.set_title("Generations vs Mutation rate")
    # plt.show()
    
    # gen_list_crossover = []
    # for crossover_rate in crossover_rates:
    #   total_iterations = 0
    #   for i in range(20):
    #     filename = "results/part1/4/crossover_rate_{}_run{}.txt".format(crossover_rate, i)
    #     gen = runGA(baseline_population_size, crossover_rate, baseline_mutation_rate, filename)
    #     total_iterations += gen
    #   gen_list_crossover.append(total_iterations / 50) # Store the average iterations
      
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(crossover_rates, gen_list_crossover)
    # ax.set_xlabel("Crossover rate")
    # ax.set_ylabel("Average number of generations to reach goal")
    # ax.set_title("Generations vs Crossover rate")
    # plt.show()

    # gen_list_population = []
    # for population_size in population_sizes:
    #   total_iterations = 0      
    #   for _ in range(20):
    #     filename = "results/part1/4/population_size_{}_run{}.txt".format(population_size, i)
    #     gen = runGA(population_size, baseline_crossover_rate, baseline_mutation_rate, filename)
    #     total_iterations += gen
    #   gen_list_population.append(total_iterations / 50) # Store the average iterations

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(population_sizes, gen_list_population)
    # ax.set_xlabel("Population size")
    # ax.set_ylabel("Average number of generations to reach goal")
    # ax.set_title("Generations vs Population size")
    # plt.show()