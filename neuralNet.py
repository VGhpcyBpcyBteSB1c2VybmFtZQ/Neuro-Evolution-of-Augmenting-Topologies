import math
import copy


class NeuralNetwork:
    def __init__(self, genome):
        # identifying the output nodes
        self.__outputNodesList = []
        nodesList = genome.getNodeGenesList()

        for i in range(0, len(nodesList)):
            if (nodesList[i].getType() == 2):
                self.__outputNodesList.append(i)
            elif (nodesList[i].getType() == 1):
                break
        # to store the results of each of the nodes during calculation
        self.__nodesResults = [0] * len(nodesList)
        # storing the connection dictionary for later use
        self.__connDict = genome.getConnectionGenesDict()
        # storing sorted keys for connections for later use
        self.__sortedKeys = sorted(list(self.__connDict.keys()))

    def __activationFunction(self, val):
        return 1 / (1 + math.exp(-4.9 * val))

    def feedForward(self, inputs):
        # put the input values into the corresponding nodes
        for i in range(0, len(inputs)):
            self.__nodesResults[i] = inputs[i]

        # start the feed forward
        for key in self.__sortedKeys:
            # skip if the connection gene is not expressed
            if (not self.__connDict[key].isExpressed()):
                continue
            inp = self.__connDict[key].getInput()
            out = self.__connDict[key].getOutput()
            # apply the activation on the incoming value only if it is not an input node
            if (inp >= len(inputs)):
                temp = self.__activationFunction(self.__nodesResults[inp])
            else:
                temp = self.__nodesResults[inp]
            self.__nodesResults[out] += temp * self.__connDict[key].getWeight()

        final_vals = []
        # iterate over the output nodes and append to result
        # after applying the activation function
        for i in self.__outputNodesList:
            final_vals.append(self.__activationFunction(self.__nodesResults[i]))

        # reset the output values of all the nodes
        for i in range(0, len(self.__nodesResults)):
            self.__nodesResults[i] = 0

        return copy.deepcopy(final_vals)