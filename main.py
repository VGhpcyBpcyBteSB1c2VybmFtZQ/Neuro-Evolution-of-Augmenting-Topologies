import genome
import neat
import random
import os
from PIL import Image
import numpy
import neuralNet

maxActivations = len(neuralNet.activationList)


def myEval(network):

    ans1 = network.feedForward([0, 0, 1])[0]
    ans2 = network.feedForward([0, 1, 1])[0]
    ans3 = network.feedForward([1, 0, 1])[0]
    ans4 = network.feedForward([1, 1, 1])[0]

    e1 = abs(ans1 - (0))
    e2 = abs(ans2 - 1.0)
    e3 = abs(ans3 - 1.0)
    e4 = abs(ans4 - (0))

    avg_err = (e1 + e2 + e3 + e4)

    if (ans1 < 0.5 and ans2 >= 0.5 and ans3 >= 0.5 and ans4 < 0.5 and False):
        return 100
    else:
        return (4 - avg_err)


node0 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node1 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node2 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node3 = genome.NodeGene(2, random.randint(0, maxActivations - 1))

con1 = genome.ConnectionGene(0, 3, random.uniform(-4, 4))
con2 = genome.ConnectionGene(1, 3, random.uniform(-4, 4))
con3 = genome.ConnectionGene(2, 3, random.uniform(-4, 4))

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
for i in range(0, 1):
    print("\n0, 0 =", network.feedForward([0, 0, 1])[0])
    print("0, 1 =", network.feedForward([0, 1, 1])[0])
    print("1, 0 =", network.feedForward([1, 0, 1])[0])
    print("1, 1 =", network.feedForward([1, 1, 1])[0])
network.printNetwork()
print("\nDone")

width = 256
height = 256

data = numpy.zeros((height, width), dtype=numpy.uint8)
for x in range(0, width):
    for y in range(0, height):
        inp1 = (x / width)
        inp2 = (y / height)
        data[y, x] = (network.feedForward([inp2, inp1, 1])[0]) * 255


img = Image.fromarray(data)
img.save('test.png')
# img.show()
os.system("pause")
