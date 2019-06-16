import genome
import neat
import random
import os
from PIL import Image
import numpy
import neuralNet
import math
import sys

maxActivations = len(neuralNet.activationList)

collage = numpy.zeros((128 * 2, 128 * 3), dtype=numpy.uint8)
iterx = 0
itery = 0


def myEval(network):

    global iterx, itery
    res = 128

    for x in range(0, res):
        for y in range(0, res):
            inp1 = (x / (res - 1) * 2 - 1)
            inp2 = (y / (res - 1) * 2 - 1)
            d = math.sqrt((inp1)**2 + (inp2)**2)
            testOutput = abs(network.feedForward([inp1, inp2, d, math.sin(10 * inp1), math.cos(10 * inp2), 1])[0]) * 255
            collage[res * itery + y, res * iterx + x] = testOutput

    iterx += 1
    if (iterx % 3 == 0):
        iterx = 0
        itery += 1
    if (itery == 2):
        iterx = 0
        itery = 0

    return [0, Image.fromarray(collage)]


node0 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node1 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node2 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node3 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node4 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node5 = genome.NodeGene(0, random.randint(0, maxActivations - 1))
node6 = genome.NodeGene(2, 0)

con1 = genome.ConnectionGene(0, 6, random.uniform(-4, 4))
con2 = genome.ConnectionGene(1, 6, random.uniform(-4, 4))
con3 = genome.ConnectionGene(2, 6, random.uniform(-4, 4))
con4 = genome.ConnectionGene(3, 6, random.uniform(-4, 4))
con5 = genome.ConnectionGene(4, 6, random.uniform(-4, 4))
con6 = genome.ConnectionGene(5, 6, random.uniform(-4, 4))

newGen1 = genome.Genome()

newGen1.addNodeGene(node0)
newGen1.addNodeGene(node1)
newGen1.addNodeGene(node2)
newGen1.addNodeGene(node3)
newGen1.addNodeGene(node4)
newGen1.addNodeGene(node5)
newGen1.addNodeGene(node6)
newGen1.addConnectionGene(con1, "0_6")
newGen1.addConnectionGene(con2, "1_6")
newGen1.addConnectionGene(con3, "2_6")
newGen1.addConnectionGene(con4, "3_6")
newGen1.addConnectionGene(con5, "4_6")
newGen1.addConnectionGene(con6, "5_6")


Algo = neat.NEAT(newGen1, 6)
network = Algo.evaluate(myEval, math.inf)

result = Image.fromarray(collage)
# result.save("Collage.png")
result.show()
sys.exit()

width = 512
height = 512

data = numpy.zeros((width, height), dtype=numpy.uint8)
for x in range(0, width):
    for y in range(0, height):
        inp1 = (x / (width - 1) * 2 - 1)
        inp2 = (y / (height - 1) * 2 - 1)
        d = math.sqrt((inp1)**2 + (inp2)**2)
        data[x, y] = (network.feedForward([inp1, inp2, d, 1])[0] + 1) / 2 * 255

network.printNetwork()


img = Image.fromarray(data)
img.save('test.png')
img.show()
os.system("pause")
