import copy
import neuralNet
import random
import genome
import math
import threading
import msvcrt

gl_Gen = None
gl_Species = None
gl_HighestFitness = None
gl_Population = None
gl_Done = False


def thread_console_writer():
    global gl_Gen
    global gl_Species
    global gl_HighestFitness
    global gl_Population
    global gl_Done

    old = None
    while (True):
        if (old != gl_Gen):
            print("Generation:", gl_Gen + 1, "\tSpecies:", gl_Species, "\tHighestFitness:", gl_HighestFitness, "\tPopulation:", gl_Population)
            old = gl_Gen
        if (gl_Done):
            return


class Specie:
    def __init__(self, id, mascot):
        self.__mascot = copy.deepcopy(mascot)
        self.__members


class NEAT:
    def __init__(self, sampleGenome, population_size):
        self.__global_specieCounter = 0  # used for giving species their IDs
        self.__population_size = population_size

        # ////////////// parameters for the neat algorithm
        self.__weight_mutation_rate = 0.8  # probability of a weight getting mutated (replaced or preturbed)
        self.__weight_change_rate = 0.1    # prob of weight getting replaced after decision for mutation (remaining prob for preturbation)
        self.__node_mutation_rate = 0.005   # prob of a connection being split in half by adding a new node
        self.__connection_mutation_rate = 0.30   # prob of a new connection being made in a child genome
        self.__c1 = 1                           # contribution of disjoint genes in genetic distance
        self.__c2 = 0                        # contribution of average weight difference in genetic distance
        self.__speciation_threshold = 6.0       # threshold to cross to be considered a different species
        self.__disable_rate = 0.75             # prob of gene being disabled if it is disabled in either of the parents
        self.__bottom_ratio = 0.0            # the bottom %age of every species that is wiped out in each generation

        self.__min_perturb = -2         # max and min perturbation amounts
        self.__max_perturb = 2

        self.__min_weight = -7           # max and min weight amounts to be assigned to new and mutant connections
        self.__max_weight = 7
        # ////////////////////////////////////////////////

        # initializing the population
        # stored in the following format
        # {specieID1 : [[fitness1, genome1], [fitness2, genome2]], specieID2 : ...}
        self.__population_members = {}
        self.__population_members[self.__global_specieCounter] = []
        for i in range(0, population_size):
            self.__population_members[self.__global_specieCounter].append([0, copy.deepcopy(sampleGenome)])

    def __getNewSpecieID(self):
        self.__global_specieCounter += 1
        return self.__global_specieCounter

    # key provider for species member sorting (returns the fitness)
    def __sortKey(self, val):
        return val[0]

    def evaluate(self, fitnessEvaluator, targetFitness=None):

        # /////////////////// Threaded writer ////////////////////
        global gl_Gen
        global gl_Species
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

            newGen = {}  # to store the next generation temporarily alongside their species
            tempMembers = []  # to store the next generation temporarily before it is classified into species
            totalOverallFitness = 0  # to store overal total fitness of the entire generation
            fitness_species = {}  # to store the total fitness of each species
            overallHighestFitness = 0
            population = 0  # to store the total population of the generation

            # loop through each of the species to calculate fitnesses, sort accordingly
            for sp in self.__population_members:
                # to store the fitness sum total of the species
                fitnessSumTotal = 0
                # loop through each member of the species
                for member in self.__population_members[sp]:
                    # create the net and calculate the fitness
                    net = neuralNet.NeuralNetwork(member[1])
                    member[0] = fitnessEvaluator(net) / len(self.__population_members[sp])
                    population += 1

                # sort the members by fitness
                self.__population_members[sp].sort(key=self.__sortKey, reverse=True)
                highestFitness = self.__population_members[sp][0][0]

                # for tracking the overall highest fitness
                if (highestFitness * len(self.__population_members[sp]) > overallHighestFitness):
                    overallHighestFitness = highestFitness * len(self.__population_members[sp])
                    bestNetwork = neuralNet.NeuralNetwork(self.__population_members[sp][0][1])

                # delete a certain %age of the bottom of the species if it has more than 3 members
                size = len(self.__population_members[sp])
                if (size > 3):
                    for i in range(size - 1, math.floor(size * (1 - self.__bottom_ratio)) - 1, -1):
                        del self.__population_members[sp][i]

                # add the best member of the species into the new generation as is
                newGen[sp] = [[0, copy.deepcopy(self.__population_members[sp][0][1])]]

            # calculating the total fitness of each species and the generation

            for sp in self.__population_members:
                # to store the fitness sum total of the species
                fitnessSumTotal = 0
                # loop through each member of the species
                for member in self.__population_members[sp]:
                    fitnessSumTotal += member[0]
                fitness_species[sp] = fitnessSumTotal
                totalOverallFitness += fitnessSumTotal

            # create the new generation by randomly breeding members within each species
            for sp in self.__population_members:
                # create total members of species proportional to its total fitness
                for total in range(0, round((fitness_species[sp] / totalOverallFitness) * (self.__population_size - len(self.__population_members)))):
                    r = random.random() * fitness_species[sp]
                    temp = 0
                    p1 = None   # parent 1
                    p2 = None   # parent 2
                    for member in self.__population_members[sp]:  # randomly choose parent 1
                        temp += member[0]
                        if (r <= temp):
                            p1 = member
                            p2 = member
                            break

                    while(p1 == p2 and len(self.__population_members[sp]) > 1):
                        r = random.random() * fitness_species[sp]
                        temp = 0
                        for member in self.__population_members[sp]:  # randomly choose parent 2
                            temp += member[0]
                            if (r <= temp):
                                p2 = member
                                break
                    # create the child and append it
                    if (p1[0] > p2[0]):
                        tempMembers.append(genome.crossover(p1[1], p2[1], self.__weight_mutation_rate, self.__weight_change_rate,
                                           self.__node_mutation_rate, self.__connection_mutation_rate, self.__disable_rate, self.__min_perturb,
                                           self.__max_perturb, self.__min_weight, self.__max_weight))
                    else:
                        tempMembers.append(genome.crossover(p2[1], p1[1], self.__weight_mutation_rate, self.__weight_change_rate,
                                           self.__node_mutation_rate, self.__connection_mutation_rate, self.__disable_rate, self.__min_perturb,
                                           self.__max_perturb, self.__min_weight, self.__max_weight))

            # put the children into their respective species
            for member in tempMembers:
                isAdded = False
                for sp in self.__population_members:
                    r = random.randrange(0, len(self.__population_members[sp]))  # choose a random mascot for the species
                    mascot = self.__population_members[sp][r][1]

                    # calculate genetic distance to the mascot
                    distance = genome.geneticDistance(member, mascot, self.__c1, self.__c2)
                    if (distance <= self.__speciation_threshold):
                        newGen[sp].append([0, member])
                        isAdded = True
                        break
                if (not isAdded):  # if offspring does not belong to any species then create a new species
                    newID = self.__getNewSpecieID()
                    newGen[newID] = [[0, member]]

            gl_Gen = gen + 1
            gl_Species = len(self.__population_members)
            gl_HighestFitness = overallHighestFitness
            gl_Population = population

            if (overallHighestFitness >= targetFitness or msvcrt.kbhit()):
                gl_Done = True
                writer.join()
                break

            del self.__population_members
            self.__population_members = copy.deepcopy(newGen)

        return copy.deepcopy(bestNetwork)
