import math
import sys


class NeuralNetwork:
    def __init__(self, genome):
        # identifying the output nodes
        self.__outputNodesList = []
        self.__nodesList = genome.getNodeGenesList()

        for i in range(0, len(self.__nodesList)):
            if (self.__nodesList[i].getType() == 2):
                self.__outputNodesList.append(i)
            elif (self.__nodesList[i].getType() == 1):
                break
        # to store the total number of nodes
        self.__totalNodes = len(self.__nodesList)
        # storing the connection dictionary for later use
        self.__connDict = genome.getConnectionGenesDict()
        # storing sorted keys for connections for later use
        self.__sortedKeys = []
        for member in self.__connDict.values():
            self.__sortedKeys.append(member)
        self.__sortedKeys.sort(key=self.__sortKey)
        for i in range(0, len(self.__sortedKeys)):
            self.__sortedKeys[i] = str(self.__sortedKeys[i].getInput()) + "_" + str(self.__sortedKeys[i].getOutput())

        # print("Nodes:", len(nodesList), "Connections: ", len(self.__connDict))

    def printNetwork(self):
        print("\n")
        print("Nodes:", self.__totalNodes)
        for key in self.__sortedKeys:
            print(key, self.__connDict[key].getWeight(), self.__connDict[key].isExpressed())

    # key provider to sort the nodesList
    def __sortKey(self, val):
        return val.getInput()

    def __activationFunction(self, val, id):
        if (id == 0):
            return math.tanh(val)      # sigmoid (-1, 1)
        elif (id == 1):
            return math.sin(val)       # sine (-1, 1)
        elif (id == 2):
            if (val > 1):              # linear bounded (-1, 1)
                return 1
            elif (val < -1):
                return -1
            else:
                return val
        elif (id == 3):
            return (math.exp(-(val * val) / (2 * 0.3 * 0.3)))   # Gaussian (0, 1)
        elif (id == 4):
            return (math.exp(-(val * val) / (2 * 4)))   # Gaussian (flatter) (0, 1)
        elif (id == 5):
            return (math.sin(val) + 1) / 2             # sine (0, 1)
        else:
            sys.exit("Activation id out of range\n")

        # if (val < 0):
        #     return 1 - (1 / (1 + math.exp(4.9 * val)))
        # else:
        #     return 1 / (1 + math.exp(-4.9 * val))

    def feedForward(self, inputs):
        # put the input values into the corresponding nodes
        nodesResults = [0] * self.__totalNodes
        for i in range(0, len(inputs)):
            nodesResults[i] = inputs[i]

        # start the feed forward
        for key in self.__sortedKeys:
            # skip if the connection gene is not expressed
            if (not self.__connDict[key].isExpressed()):
                continue
            inp = self.__connDict[key].getInput()
            out = self.__connDict[key].getOutput()
            # apply the activation on the incoming value only if it is not an input node
            if (self.__nodesList[inp].getType() != 0):
                temp = self.__activationFunction(nodesResults[inp], self.__nodesList[inp].getActivation())
            else:
                temp = nodesResults[inp]
            nodesResults[out] += temp * self.__connDict[key].getWeight()

        final_vals = []
        # iterate over the output nodes and append to result
        # after applying the activation function
        for i in self.__outputNodesList:
            final_vals.append(self.__activationFunction(nodesResults[i], self.__nodesList[i].getActivation()))

        return final_vals
