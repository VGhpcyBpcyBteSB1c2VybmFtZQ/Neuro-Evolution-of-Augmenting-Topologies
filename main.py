import genome
import neat


def myEval(network):
    e1 = abs(network.feedForward([0, 0, 1])[0] - 0.0)
    e2 = abs(network.feedForward([0, 1, 1])[0] - 1.0)
    e3 = abs(network.feedForward([1, 0, 1])[0] - 1.0)
    e4 = abs(network.feedForward([1, 1, 1])[0] - 0.0)

    avg_err = (e1 + e2 + e3 + e4)

    return (4 - avg_err)**2


node0 = genome.NodeGene(0)
node1 = genome.NodeGene(0)
node2 = genome.NodeGene(0)
node3 = genome.NodeGene(2)

con1 = genome.ConnectionGene(0, 3, -1.6)
con2 = genome.ConnectionGene(1, 3, 0.9)
con3 = genome.ConnectionGene(2, 3, 0.2)

newGen1 = genome.Genome()

newGen1.addNodeGene(node0)
newGen1.addNodeGene(node1)
newGen1.addNodeGene(node2)
newGen1.addNodeGene(node3)
newGen1.addConnectionGene(con1, "0_3")
newGen1.addConnectionGene(con2, "1_3")
newGen1.addConnectionGene(con3, "2_3")

Algo = neat.NEAT(newGen1, 20)
network = Algo.evaluate(myEval, 60)

for i in range(0, 1):
    print("\n0, 0 =", network.feedForward([0, 0, 1])[0])
    print("0, 1 =", network.feedForward([0, 1, 1])[0])
    print("1, 0 =", network.feedForward([1, 0, 1])[0])
    print("1, 1 =", network.feedForward([1, 1, 1])[0])

network.printNetwork()
