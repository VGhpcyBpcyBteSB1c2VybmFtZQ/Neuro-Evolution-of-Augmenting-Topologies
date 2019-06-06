import genome
import neat


def myEval(network):
    e1 = abs(network.feedForward([0, 0])[0] - 0)
    e2 = abs(network.feedForward([0, 1])[0] - 0)
    e3 = abs(network.feedForward([1, 0])[0] - 0)
    e4 = abs(network.feedForward([1, 1])[0] - 1)

    avg_err = (e1 + e2 + e3 + e4) / 4

    return (1 / (1 + avg_err))


node1 = genome.NodeGene(0)
node2 = genome.NodeGene(0)
node3 = genome.NodeGene(2)

con1 = genome.ConnectionGene(0, 2, -1.6)
con2 = genome.ConnectionGene(1, 2, 0.9)

newGen1 = genome.Genome()
newGen2 = genome.Genome()

newGen1.addNodeGene(node1)
newGen1.addNodeGene(node2)
newGen1.addNodeGene(node3)
newGen1.addConnectionGene(con1, "0_2")
newGen1.addConnectionGene(con2, "1_2")

Algo = neat.NEAT(newGen1, 10)
Algo.evaluate(myEval, 1000)
