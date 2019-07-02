import genome
import neat
import random
import os
from PIL import Image
import numpy as np
import neuralNet

maxActivations = len(neuralNet.activationList)


def myEval(network):

    inp1 = np.array([0, 0, 1, 1])
    inp2 = np.array([0, 1, 0, 1])
    inp3 = np.array([1, 1, 1, 1])

    expected = np.array([0, 1, 1, 0])

    result = np.sum(np.absolute(network.feedForward([inp1, inp2, inp3])[0] - expected))

    return (4 - result)


node0 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node1 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node2 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node3 = genome.NodeGene(2, random.randint(0, maxActivations - 1))

# con1 = genome.ConnectionGene(0, 3, random.uniform(-4, 4))
# con2 = genome.ConnectionGene(1, 3, random.uniform(-4, 4))
# con3 = genome.ConnectionGene(2, 3, random.uniform(-4, 4))

newGen1 = genome.Genome()

newGen1.addNodeGene(node0)
newGen1.addNodeGene(node1)
newGen1.addNodeGene(node2)
newGen1.addNodeGene(node3)
# newGen1.addConnectionGene(con1, "0_3")
# newGen1.addConnectionGene(con2, "1_3")
# newGen1.addConnectionGene(con3, "2_3")

Algo = neat.NEAT(newGen1, 150)
network = Algo.evaluate(myEval, 4)

inp1 = np.array([0, 0, 1, 1])
inp2 = np.array([0, 1, 0, 1])
inp3 = np.array([1, 1, 1, 1])

result = network.feedForward([inp1, inp2, inp3])[0].reshape((4, 1))
print("\n")
print(result)
network.printNetwork()
print("\nDone")

dimension = 256

inp1 = np.arange(dimension).reshape((1, dimension)).repeat(dimension, axis=0) / dimension
inp2 = np.transpose(inp1)
inp3 = np.ones((dimension, dimension))

data = (network.feedForward([inp2, inp1, inp3])[0] * 255).astype(dtype=np.uint8)


img = Image.fromarray(data)
img.save('test.png')
# img.show()
os.system("pause")
