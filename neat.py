import copy
import neuralNet
import random
import genome
import math
import threading
import msvcrt

gl_Gen = None
gl_HighestFitness = None
gl_Population = None
gl_Done = False


def thread_console_writer():
    global gl_Gen
    global gl_HighestFitness
    global gl_Population
    global gl_Done

    old = None
    while (True):
        if (old != gl_Gen):
            print("Generation:", gl_Gen + 1, "\tHighestFitness:", gl_HighestFitness, "\tPopulation:", gl_Population)
            old = gl_Gen
        if (gl_Done):
            return


class NEAT:
    def __init__(self, sampleGenome, population_size):
        self.__population_size = population_size

        # ////////////// parameters for the neat algorithm
        self.__weight_mutation_rate = 0.8  # probability of a weight getting mutated (replaced or preturbed)
        self.__weight_change_rate = 0.1    # prob of weight getting replaced after decision for mutation (remaining prob for preturbation)
        self.__node_mutation_rate = 0.005   # prob of a connection being split in half by adding a new node
        self.__connection_mutation_rate = 0.01   # prob of a new connection being made in a child genome
        self.__disable_rate = 0.75             # prob of gene being disabled if it is disabled in either of the parents
        self.__bottom_ratio = 0            # the bottom %age of every species that is wiped out in each generation

        self.__min_perturb = -2         # max and min perturbation amounts
        self.__max_perturb = 2

        self.__min_weight = -7           # max and min weight amounts to be assigned to new and mutant connections
        self.__max_weight = 7
        # ////////////////////////////////////////////////

        # initializing the population
        # stored in the following format
        # [[fitness1, genome1], [fitness2, genome2].....]
        self.__population_members = []
        for i in range(0, population_size):
            self.__population_members.append([0, copy.deepcopy(sampleGenome)])

    # key provider for species member sorting (returns the fitness)
    def __sortKey(self, val):
        return val[0]

    def evaluate(self, fitnessEvaluator, targetFitness=None):

        # /////////////////// Threaded writer ////////////////////
        global gl_Gen
        global gl_HighestFitness
        global gl_Population
        global gl_Done
        gl_Done = False

        writer = threading.Thread(target=thread_console_writer, args=())
        writer.start()
        # ////////////////////////////////////////////////////////

        bestNetwork = None
        gen = -1

        while(True):
            gen += 1
            random.seed()

            tempMembers = []  # to store the next generation temporarily
            totalGenerationFitness = 0
            overallHighestFitness = 0
            population = 0  # to store the total population of the generation

            # loop through each member to calculate fitnesses, sort accordingly
            for member in self.__population_members:
                net = neuralNet.NeuralNetwork(member[1])
                member[0] = fitnessEvaluator(net)
                population += 1
                totalGenerationFitness += member[0]

            # sort the members by fitness
            self.__population_members.sort(key=self.__sortKey, reverse=True)
            overallHighestFitness = self.__population_members[0][0]
            bestNetwork = neuralNet.NeuralNetwork(self.__population_members[0][1])

            # delete a certain %age of the bottom of the species if it has more than 3 members
            size = len(self.__population_members)
            if (size > 3):
                for i in range(size - 1, math.floor(size * (1 - self.__bottom_ratio)) - 1, -1):
                    del self.__population_members[i]

            # add the best member into the new generation as is
            tempMembers = [self.__population_members[0]]

            # create the new generation by randomly breeding members within each species
            for iterator in range(0, self.__population_size - 1):
                r = random.random() * totalGenerationFitness
                temp = 0
                p1 = None   # parent 1
                p2 = None   # parent 2
                for member in self.__population_members:  # randomly choose parent 1
                    temp += member[0]
                    if (r <= temp):
                        p1 = member
                        p2 = member
                        break

                while(p1 == p2 and len(self.__population_members) > 1):
                    r = random.random() * totalGenerationFitness
                    temp = 0
                    for member in self.__population_members:  # randomly choose parent 2
                        temp += member[0]
                        if (r <= temp):
                            p2 = member
                            break
                # create the child and append it
                if (p1[0] > p2[0]):
                    tempMembers.append([0, genome.crossover(p1[1], p2[1], self.__weight_mutation_rate, self.__weight_change_rate,
                                       self.__node_mutation_rate, self.__connection_mutation_rate, self.__disable_rate, self.__min_perturb,
                                       self.__max_perturb, self.__min_weight, self.__max_weight)])
                else:
                    tempMembers.append([0, genome.crossover(p2[1], p1[1], self.__weight_mutation_rate, self.__weight_change_rate,
                                       self.__node_mutation_rate, self.__connection_mutation_rate, self.__disable_rate, self.__min_perturb,
                                       self.__max_perturb, self.__min_weight, self.__max_weight)])

            gl_Gen = gen + 1
            gl_HighestFitness = overallHighestFitness
            gl_Population = population

            if (overallHighestFitness >= targetFitness or msvcrt.kbhit()):
                gl_Done = True
                writer.join()
                break

            del self.__population_members
            self.__population_members = tempMembers

        return copy.deepcopy(bestNetwork)
