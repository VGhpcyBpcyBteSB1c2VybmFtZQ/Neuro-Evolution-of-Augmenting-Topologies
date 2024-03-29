import sys
import random
import copy

# NodeGene: used to represent the node of the neural net, has an id and a type
# type 0 = input
# type 1 = hiden
# type 2 = output


class NodeGene:
    def __init__(self, type):
        self.__type = type

    def getType(self):
        return self.__type

    def setType(self, type):
        self.__type = type

# ConnectionGene: used to represent the connection between nodes


class ConnectionGene:
    def __init__(self, input, output, weight):
        self.__input = input
        self.__output = output
        self.__weight = weight
        self.__expressed = True

    def getInput(self):
        return self.__input

    def getOutput(self):
        return self.__output

    def getWeight(self):
        return self.__weight

    def setWeight(self, weight):
        self.__weight = weight

    def enable(self):
        self.__expressed = True

    def disable(self):
        self.__expressed = False

    def isExpressed(self):
        return self.__expressed

# Genome: used to represent the genome of an indivdual in a population


class Genome:
    def __init__(self):
        self.__nodeGene_list = []
        self.__connectionGene_dict = {}

    def addNodeGene(self, nodeGene):
        self.__nodeGene_list.append(copy.deepcopy(nodeGene))

    def addConnectionGene(self, connectionGene, innovation):  # the innovation is of the form "1_2" where 1 is in and 2 is out
        if (innovation in self.__connectionGene_dict):
            sys.exit("Innovation already present in genome\n")
        self.__connectionGene_dict[innovation] = copy.deepcopy(connectionGene)

    def getNodeGenesList(self):
        return copy.deepcopy(self.__nodeGene_list)

    def getConnectionGenesDict(self):
        return copy.deepcopy(self.__connectionGene_dict)


# function for crossing over two genomes, the first parent is assumed to have higher fitness
def crossover(genome1, genome2, weight_mutation_rate, weight_change_rate, node_mutation_rate, connection_mutation_rate,
              disable_rate, min_perturb, max_perturb, min_weight, max_weight):
    random.seed()
    newGenome = Genome()

    # add all the node genes from the parent with the higher fitness
    nodesGenome = genome1.getNodeGenesList()
    for node in nodesGenome:
        newGenome.addNodeGene(node)

    connectionsGenome1 = genome1.getConnectionGenesDict()
    connectionsGenome2 = genome2.getConnectionGenesDict()
    for i in connectionsGenome1:
        if (i in connectionsGenome2):
            # choosing a random gene from either of the parents
            prob = random.randint(1, 2)
            if (prob == 1):
                conGene = connectionsGenome1[i]
            else:
                conGene = connectionsGenome2[i]
        else:
            # choose the gene from the highest fitness parent
            conGene = connectionsGenome1[i]

        # mutating the weight (replacing it or perturbing it)
        prob = random.random()
        if (prob <= weight_mutation_rate):
            prob = random.random()
            if (prob <= weight_change_rate):
                conGene.setWeight(random.uniform(min_weight, max_weight))
            else:
                conGene.setWeight(conGene.getWeight() + random.uniform(min_perturb, max_perturb))

        # disabling a gene if it is disabled in either of the parents 75% of the times
        if (i in connectionsGenome1 and (not connectionsGenome1[i].isExpressed())):
            disabled = True
        elif (i in connectionsGenome2 and (not connectionsGenome2[i].isExpressed())):
            disabled = True
        else:
            disabled = False
        if (disabled):
            prob = random.random()
            if (prob <= disable_rate):
                conGene.disable()
            else:
                conGene.enable()

        # mutating the connection by adding a new node and splitting it
        prob = random.random()
        if (prob <= node_mutation_rate):
            conGene.disable()
            newInnovation1 = str(conGene.getInput()) + "_" + str(len(newGenome.getNodeGenesList()))
            newInnovation2 = str(len(newGenome.getNodeGenesList())) + "_" + str(conGene.getOutput())

            conGene1 = ConnectionGene(conGene.getInput(), len(newGenome.getNodeGenesList()), conGene.getWeight())
            conGene2 = ConnectionGene(len(newGenome.getNodeGenesList()), conGene.getOutput(), 1)

            nodeGene = NodeGene(1)

            newGenome.addNodeGene(nodeGene)
            # print("Adding inno1 into newGenome")
            newGenome.addConnectionGene(conGene1, newInnovation1)
            # print("newInnovation1", newInnovation1)
            # print("Adding inno2 into newGenome:", newInnovation1, newInnovation2)
            newGenome.addConnectionGene(conGene2, newInnovation2)
            # print("newInnovation2", newInnovation2)

        # print("i", i)
        # print("Adding conGene into newGenome")
        newGenome.addConnectionGene(conGene, i)  # adding the gene into the child

    # mutating by creating a previously absent connection
    prob = random.random()
    if (prob <= connection_mutation_rate):
        # try choosing a random connection following number of times and quit if can't find it
        connectionsNewGenome = newGenome.getConnectionGenesDict()
        nodeList = newGenome.getNodeGenesList()
        for f in range(0, len(connectionsNewGenome)):
            start = random.randrange(0, len(nodeList))
            end = random.randrange(0, len(nodeList))
            if (start != end and (nodeList[start].getType() != 2) and (start < end)):
                if (nodeList[start].getType() != 0 or nodeList[end].getType() != 0):
                    if (nodeList[start].getType() != 2 or nodeList[end].getType() != 2):
                        inno = str(start) + "_" + str(end)
                        if(inno not in connectionsNewGenome):
                            mutantCon = ConnectionGene(start, end, random.uniform(min_weight, max_weight))
                            #print("Adding absent conn into newGenome")
                            newGenome.addConnectionGene(mutantCon, inno)
                            break

    return copy.deepcopy(newGenome)


# function to calculate the genetic distance between two genomes
def geneticDistance(genome1, genome2, c1, c2):
    g1_p1 = genome1.getConnectionGenesDict()
    g2_p1 = genome2.getConnectionGenesDict()

    if (len(g1_p1) > len(g2_p1)):
        N = len(g1_p1)
    else:
        N = len(g2_p1)

    if (N < 20):
        N = 1

    total_disjoint_genes = 0
    average_weight_difference = 0
    total_common_connections = 0

    for inno in g1_p1:
        if inno in g2_p1:
            average_weight_difference += abs(g1_p1[inno].getWeight() - g2_p1[inno].getWeight())
            total_common_connections += 1
        else:
            total_disjoint_genes += 1

    for inno in g2_p1:
        if inno not in g1_p1:
            total_disjoint_genes += 1

    average_weight_difference /= total_common_connections
    return (c1 * (total_disjoint_genes / N)) + (c2 * average_weight_difference)
