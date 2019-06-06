import copy
import neuralNet
import random
import genome
import math


class Specie:
    def __init__(self, id, mascot):
        self.__mascot = copy.deepcopy(mascot)
        self.__members


class NEAT:
    def __init__(self, sampleGenome, population_size):
        self.__global_specieCounter = 0  # used for giving species their IDs
        self.__population_size = population_size

        # ////////////// parameters for the neat algorithm
        self.__weight_mutation_rate = 0.8
        self.__weight_change_rate = 0.1
        self.__node_mutation_rate = 0.03
        self.__connection_mutation_rate = 0.05
        self.__c1 = 1
        self.__c2 = 0.3
        self.__speciation_threshold = 3.0
        # ////////////////////////////////////////////////

        # stores the number of generations a specie has stayed stagnant
        # stores in following format
        # {specieID : [highestFitness, numberOfGenerations]}
        self.__specieFitnessStagnation = {self.__global_specieCounter: [0, 0]}

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

    def evaluate(self, fitnessEvaluator, generations):
        sterilized = []  # stores the species not allowed to reproduce

        for gen in range(0, generations):

            newGen = {}  # to store the next generation temporarily alongside their species
            tempMembers = []  # to store the next generation temporarily before it is classified into species
            totalOverallFitness = 0  # to store overal total fitness of the entire generation
            fitness_species = {}  # to store the total fitness of each species
            overallHighestFitness = 0

            # loop through each of the species to calculate fitnesses, sort accordingly
            # and remove stagnant species
            for sp in self.__population_members:
                # to store the fitness sum total of the species
                fitnessSumTotal = 0
                # loop through each member of the species
                for member in self.__population_members[sp]:
                    # create the net and calculate the fitness
                    net = neuralNet.NeuralNetwork(member[1])
                    member[0] = fitnessEvaluator(net) / len(self.__population_members[sp])

                # sort the members by fitness and check for stagnation
                self.__population_members[sp].sort(key=self.__sortKey, reverse=True)
                highestFitness = self.__population_members[sp][0][0]

                # for tracking the overall highest fitness
                if (highestFitness > overallHighestFitness):
                    overallHighestFitness = highestFitness

                if (highestFitness <= self.__specieFitnessStagnation[sp][0]):
                    self.__specieFitnessStagnation[sp][1] += 1
                else:
                    self.__specieFitnessStagnation[sp][0] = highestFitness
                    self.__specieFitnessStagnation[sp][1] = 0

                # if species has been stagnant for >= 15 generations, then add it to sterilized list and add as is to
                # the next generation
                if (self.__specieFitnessStagnation[sp][1] >= 15):
                    sterilized.append(sp)
                    newGen[sp] = []
                    for i in range(0, len(self.__population_members[sp])):
                        newGen[sp].append([0, copy.deepcopy(self.__population_members[sp][i][1])])
                else:
                    # delete the bottom 50% of the species if it has more than 2 members
                    size = len(self.__population_members[sp])
                    if (size > 2):
                        for i in range(size - 1, math.floor(size / 2) - 1, -1):
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
                # skip if species is sterile
                if sp in sterilized:
                    continue
                # create total members of species proportional to its total fitness
                for total in range(0, math.floor(fitness_species[sp] / totalOverallFitness * self.__population_size)):
                    r = random.random() * fitness_species[sp]
                    temp = 0
                    p1 = None   # parent 1
                    p2 = None   # parent 2
                    for member in self.__population_members[sp]:  # randomly choose parent 1
                        temp += member[0]
                        if (r <= temp):
                            p1 = member[1]
                            p2 = member[1]

                    while(p1 == p2 and len(self.__population_members[sp]) > 1):
                        r = random.random() * fitness_species[sp]
                        temp = 0
                        for member in self.__population_members[sp]:  # randomly choose parent 2
                            temp += member[0]
                            if (r <= temp):
                                p2 = member[1]
                                break

                    # create the child and append it
                    tempMembers.append(genome.crossover(p1, p2, self.__weight_mutation_rate, self.__weight_change_rate, self.__node_mutation_rate, self.__connection_mutation_rate))

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
                    self.__specieFitnessStagnation[newID] = [0, 0]

            print("Generation:", gen + 1, "Species:", len(self.__population_members), ", Highest Fitness:", overallHighestFitness)
            del self.__population_members
            self.__population_members = copy.deepcopy(newGen)
