import math
import sys


# /////////////////////// ACTIVATION FUNCTIONS FOR CPPN ////////////////////////////


def sigmoid(val):
    return math.tanh(val)


def sigmoidBounded(val):
    return (math.tanh(val) + 1) / 2


def sine(val):
    return math.sin(val)


def linearBounded(val):
    if (val > 1):
        return 1
    elif (val < 0):
        return 0
    else:
        return val


def step(val):
    if (val > 0):
        return 1
    elif (val < 0):
        return 0
    else:
        return val


def gaussian(val):
    return (math.exp(-(val * val) / (2 * 0.3 * 0.3)))


def gaussianFlatter(val):
    return (math.exp(-(val * val) / (2 * 4)))


def sineBounded(val):
    return (math.sin(val) + 1) / 2


def cosine(val):
    return math.cos(val)


def cosineBounded(val):
    return (math.cos(val) + 1) / 2


activationList = [sigmoidBounded, sineBounded, linearBounded, step, gaussianFlatter]

# //////////////////////////////////////////////////////////////////////////////////


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
        if (id < len(activationList)):
            return activationList[id](val)
        else:
            sys.exit("Activation id out of range\n")

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
