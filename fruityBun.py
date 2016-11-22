import sys, random, math
from collections import namedtuple
import matplotlib.pyplot as plt
import scipy.cluster.vq as clustering
import numpy as np
from scipy.cluster.vq import kmeans, vq
import threading
import gevent
from gevent import Greenlet
from multiprocessing import Process, Manager, Pool
from threading import Thread

from cluster import Cluster
from coordinates import Coordinates
from deposit import Deposit
from permutation import Permutation
from truck import Truck

from helperFunctions import *

top_solutions   = 3
MUTATION_RATIO  = 201

NR_SOLUTIONS    = 25
ITERATIONS      = 1000

INCREASE_SOLUTIONS   = 0
INCREASE_ITERATIONS  = 9000

TRUCKS          = 25
PROCESSORS      = 4
CHANGES         = 2
THREADS         = 100
REPLACE         = 20
NR_MUTATIONS    = 1
MUTATION_RATE   = 0.8


CrossPair   = namedtuple("CrossPair", "sol1, sol2")

def combineSolutions(s1, s2):
    global CHANGES
    scopy1 = Permutation(s1.deposit, s1.destinations[:], s1.truck_capacity)
    scopy2 = Permutation(s2.deposit, s2.destinations[:], s2.truck_capacity)

    # for _ in range(CHANGES):
    s1_client_1, s1_client_2  = s1.findCrossOverPoint()
    scopy2.linkDestinatios(s1_client_1, s1_client_2)

    # for _ in range(CHANGES):
    s2_client_1, s2_client_2  = s2.findCrossOverPoint()
    scopy1.linkDestinatios(s2_client_1, s2_client_2)

    return scopy1,scopy2

def combineSolutions2(s1, s2, clusterMatrix):
    c1 = random.randint(0, 23)
    c2 = random.randint(0, 23)
    # print c1, c2
    scopy1 = Permutation(s1.deposit, s1.destinations[:], s1.truck_capacity)
    scopy2 = Permutation(s2.deposit, s2.destinations[:], s2.truck_capacity)

    # c1_n = scopy1.getClusterNeighbours(c1)
    # c2 = clusterMatrix[c1_n[-1]][random.randint(1,3)]
    # c2 = random.randint(0,23)
    scopy1.swapClusters(c1, c2)

    # c1_n = scopy2.getClusterNeighbours(c1)
    # c2 = clusterMatrix[c1_n[-1]][random.randint(1,3)]
    scopy2.swapClusters(c1, c2)

    return scopy1, scopy2

# uniform ordered permutation
def UOX(s1, s2):
    length = len(s1.destinations)
    bits = ('{0:0'+str(length) + 'b}').format(random.getrandbits(length))

    scopy1 =  Permutation(s1.deposit, s1.destinations[:], s1.truck_capacity)
    start_index = None
    elements    = []
    # while string not
    for i in range(0, len(bits)):
        bit = int(bits[i])

        if bit :
            if start_index is None :
                start_index = i

            elements.append(s2.destinations[i])
        else :
            scopy1.insertElements(start_index, elements)
            start_index = None
            elements = []
    return scopy1

def chooseSolution(sol1, sol2):
    if sol1.fitness > sol2.fitness :
        return sol2
    else:
        return sol1

# crossover between 2 solution
def crossOver(solutions, clusterMatrix):
    pairs   = []
    results = []
    while(len(solutions) > 1):
        first = solutions[random.randint(0, len(solutions)-1)]
        solutions.remove(first)

        second = solutions[random.randint(0, len(solutions)-1)]
        solutions.remove(second)

        pairs.append(CrossPair(first, second))

    for i in range(0, len(pairs)):
        first, second = combineSolutions2(pairs[i].sol1, pairs[i].sol2, clusterMatrix)
        first = chooseSolution(first, pairs[i].sol1)
        second = chooseSolution(second, pairs[i].sol2)
        for _ in range(CHANGES):
            first1, second1 = combineSolutions(first, second)
            first = chooseSolution(first, first1)
            second = chooseSolution(second, second1)
        solutions.extend([first, second])

def mutate(s1, clusterMatrix):
    if random.random() < MUTATION_RATE :
        # scopy1 = Permutation(s1.deposit, s1.destinations[:], s1.truck_capacity)
        for _ in range(NR_MUTATIONS):
            # scopy1 = chooseSolution(scopy1, s1)
            c1 = random.randint(0, 23)
            c1_n = s1.getClusterNeighbours(c1)
            # c2 = clusterMatrix[c1_n[-1]][1]
            c2 = clusterMatrix[c1_n[random.randint(0, len(c1_n)-1)]][random.randint(1, 10)]
            # c2 = random.randint(1, 23)
            s1.swapClusters(c1, c2)

    # return chooseSolution(s1, scopy1)

def mutations(solutions, clusterMatrix) :
    for _ in range(len(solutions)):
        index = random.randint(0, len(solutions)-1)
        mutate(solutions[index],clusterMatrix)

def get_labels(deposits):
    coords = map(lambda dep: [float(dep.position.x), float(dep.position.y)] , deposits)
    y = np.array(coords)
    codebook, _ = kmeans(y, 25)  # three clusters
    cluster_indeces, _ = vq(y, codebook)

    xpoints = [str(d.position.x) for d in deposits]
    ypoints = [str(d.position.y) for d in deposits]

    for i in range(0, len(cluster_indeces)):
        deposits[i].label = cluster_indeces[i]

    return cluster_indeces

def posDist(location1, location2):
    return math.sqrt( pow(location1.x - location2.x, 2) + pow(location1.y - location2.y, 2) )

def get_clusters(deposits, labels):
    matrix = []
    for d in deposits:
        while len(matrix) <= d.label  :
            matrix.append([])
        matrix[d.label].append(d.position)

    clusters = []
    index = 0
    for row in matrix:
        clusters.append(Cluster(index, row))
        index += 1
    return clusters

def createTableRow(clusters, index, matrix):
    row = matrix[index]
    order = []
    for i in range(0, len(matrix)):
        if len(order) == 0 :
            order.append(i)
        else:
            j = 0
            while j < len(order) and row[i] > row[order[j]] :
                j +=1
            if j == len(order):
                order.append(i)
            else :
                order.insert(j, i)

    return order

def get_clusters_matrix(destinations, labels):
    clusters  = get_clusters(destinations, labels)
    matrix  = [ ]
    for i in range(0, len(clusters)):
        # print clusters[i].centre
        matrix.append([])
        for j in range(0, len(clusters)):
            matrix[i].append(posDist(clusters[i].centre, clusters[j].centre))

    lookUpTable = matrix[:]
    for i in range(0, len(clusters)):
        lookUpTable[i] = createTableRow(clusters, i, matrix)

    return lookUpTable

def generateMultipleSolutions(deposits, capacity, nr_solutions, labels=None, clustersMatrix=None):
    global PROCESSORS

    if nr_solutions < PROCESSORS :
        PROCESSORS = nr_solutions

    # start with no solution
    solutions   = []

    # get labels of clusters for each deposit
    if labels is None :
        labels = get_labels(deposits)

    if clustersMatrix is None :
        clustersMatrix = get_clusters_matrix(deposits, labels)

    manager = Manager()
    solution = manager.dict()

    randomSolution = []
    for i in range(0,PROCESSORS):
        randomSolution.append(0)

    while len(solutions) < nr_solutions :
        if nr_solutions - len(solutions) < PROCESSORS :
            PROCESSORS =  nr_solutions - len(solutions)

        for pid in range(0, PROCESSORS):
            temp_deposits       = deposits[:]
            randomSolution[pid]   = Process(target = generateRandomSolution, args=(temp_deposits, capacity, solution, pid, labels, clustersMatrix) )
            randomSolution[pid].start()
        for pid in range(0,PROCESSORS):
            randomSolution[pid].join()
            solutions.append(solution[pid])

    return solutions, labels, clustersMatrix

def generateRandomSolution(deposits, capacity, solution, sol_index, depo_labels = [], clustersMatrix = []):
    global TRUCKS
    trucks = []
    destinations = []
    startDeposit = deposits[0];
    deposits.remove(deposits[0])

    # initialise all trucks with start deposit
    for i in range(0, TRUCKS):
        trucks.append(Truck(capacity, startDeposit))

    if len(depo_labels) == 0 :
        while(len(deposits) > 0):
            allocated = False
            index = random.randint(0, len(deposits)-1)
            destinations.append(deposits[index])
            deposits.remove(deposits[index])
    else:
        failed = []
        while len(deposits) > 0 :
            allocated = False
            index = random.randint(0, len(deposits)-1)
            label = deposits[index].label
            if trucks[label].capacity >= deposits[index].demand:
                trucks[label].addDestination(deposits[index])
                allocated = True
            else:
                failed.append(deposits[index])

            deposits.remove( deposits[index] )
        extra = 0
        while len(failed) > 0 :
            attempt = False
            for i in range(1, len(clustersMatrix[failed[0].label])):
                trucks[clustersMatrix[failed[0].label][i]].addDestination(failed[0])
                failed.remove(failed[0])
                attempt = True
                break

            if not attempt :
                extra +=1
                print extra
                trucks.append(Truck(capacity, startDeposit))
                trucks[-1].addDestination(failed[0])
                failed.remove(failed[0])

        # print extra, len(failed), len(trucks)
    while trucks :
        index = random.randint(0, len(trucks)-1)
        destinations.extend(trucks[index].route[1:-1])
        del trucks[index]

    solution[sol_index] = Permutation(startDeposit, destinations, capacity)




def getParents(solutions):
    mergeSort(solutions)
    size = len(solutions)
    parentsRatio = 0.4
    parents = []
    if size == 1 :
        return []

    while len(solutions) > (1 - parentsRatio)*size :
        index = random.randint(0, int((1-parentsRatio)*size)-1)
        p = solutions[index]
        parents.append(p)
        del solutions[index]
    return parents

def getCrossRange(s):
    pos1 = random.randint(0, int(len(s.destinations)-len(s.destinations)/10-1))
    length = random.randint(0, int(len(s.destinations)/10))
    pos2 = pos1 + length
    if pos1 > pos2 :
        return pos2, pos1
    return pos1, pos2

def simpleMutation(s):
    if random.random() < MUTATION_RATE :
        for _ in range(NR_MUTATIONS):
            pos1 = random.randint(0, len(s.destinations)-1)
            pos2 = random.randint(0, len(s.destinations)-1)
            if random.randint(0,1):
                s.invert(pos1, pos2)
            else:
                s.swapNodes(pos1, pos2)

def createOffspring(s1, s2, pos1, pos2):

    d1 = s1.destinations
    d2 = s2.destinations

    hashDest = [False]*len(d2)
    newDest = [None] * len(d2)

    for i in range(pos1, pos2):
        newDest[i] = d1[i]
        hashDest[d1[i].number-1] = True


    index = 0
    i = 0
    while index < len(newDest) :
        while i < len(d2) and hashDest[d2[i].number-1] :
            i += 1

        while index < len(newDest) and not newDest[index] is None :
            index+=1

        if i >= len(d2) or index >= len(newDest):
            break

        if(index < len(newDest) and i<len(d2)):
            newDest[index] = d2[i]
        index += 1
        i += 1

    child = Permutation(s2.deposit, newDest, s2.truck_capacity)

    return child


def orderedCrossOver(s1, s2):
    pos1, pos2 = getCrossRange(s1)
    kid1 = createOffspring(s1, s2, pos1, pos2)
    kid2 = createOffspring(s2, s1, pos1, pos2)
    return kid1, kid2

def addOffspring(kid, solutions, check):
    worst = solutions[0].fitness
    idx = 0
    for i in range(0, len(solutions)):
        if solutions[i].fitness > worst :
            worst = solutions[i].fitness
            idx = i
        if kid.fitness == solutions[i].fitness:
            return solutions

    if solutions[idx].fitness > kid.fitness or check == 0:

        solutions[idx] = kid

    return solutions

def crossParentsPairs(solutions, clusterMatrix):
    pair = []
    to_replace = []
    kids = []
    for i in range(0, len(solutions)/2):
        idx1 = random.randint(0, len(solutions)-1)
        idx2 = random.randint(0, len(solutions)-1)

        if solutions[idx1].fitness > solutions[idx2].fitness:
            pair.append(solutions[idx2])
            to_replace.append(idx1)
        else:
            pair.append(solutions[idx1])
            to_replace.append(idx2)

        if len(pair) == 2:
            kid1, kid2 = orderedCrossOver(pair[0], pair[1])
            simpleMutation(kid1)
            simpleMutation(kid2)
            # mutate(kid1, clusterMatrix)
            # mutate(kid2, clusterMatrix)
            kids.append(kid1)
            kids.append(kid2)
            # solutions[to_replace[-1]] = kid1
            # solutions[to_replace[-2]] = kid2
            # solutions = addOffspring(kid1, solutions, 0)
            # solutions = addOffspring(kid2, solutions, 0)
            pair = []
            # to_replace = []

    for i in range(0, len(kids)):
        # if solutions[to_replace[i]].fitness > kids[i].fitness:
        solutions[to_replace[i]] = kids[i]
        return solutions

def pickParent(solutions):
    sumSol = sum(x.fitness for x in solutions)
    if len(solutions) == 1:
        return 0

    probabilities = map(lambda x: (1 - x.fitness/sumSol)/(len(solutions) - 1), solutions)

    cumsums = np.cumsum(probabilities)

    something = random.random()

    for index, value in enumerate(cumsums):
        if index == len(cumsums)-1:
            return index
        if value < something and something < cumsums[index+1]:
            return index


def crossParentsNew(solutions, clusterMatrix):

    sol_copy = solutions[:]
    while len(solutions) > 2:
        idx = pickParent(solutions)
        parent1 = solutions[idx]
        del solutions[idx]

        idx2 = pickParent(solutions)
        parent2 = solutions[idx2]
        del solutions[idx2]

        kid1, kid2 = orderedCrossOver(parent1, parent2)
        simpleMutation(kid1)
        simpleMutation(kid2)
        sol_copy = addOffspring(kid1, sol_copy, idx%2)
        sol_copy = addOffspring(kid2, sol_copy, idx%2)

    return sol_copy


def crossParents(solutions, clusterMatrix):
    sort = False

    if sort :
        mergeSort(solutions)

    for i in range(0, len(solutions)):
        idx1 = random.randint(0, len(solutions)-1)
        idx2 = random.randint(0, len(solutions)-1)
        parent1 = solutions[idx1]
        parent2 = solutions[idx2]

        # kid1 = solutions[0]
        # kid2 = solutions[1]
        kid1, kid2 = orderedCrossOver(parent1, parent2)
        simpleMutation(kid1)
        simpleMutation(kid2)
        # mutate(kid1, clusterMatrix)
        # mutate(kid2, clusterMatrix)
        solutions = addOffspring(kid1, solutions, idx2%2)
        solutions = addOffspring(kid2, solutions, idx1%2)

    return solutions

def getBestNeighbour(current, deposits):
    minDistance = dist(current, deposits[0])
    best = deposits[0]

    for d in deposits:
        distance = dist(current, d)
        if distance < minDistance:
            best = d
            minDistance = distance

    return best

class Generation:
    def __init__(self, pop_nr, iterations_nr, deposits, capacity):
        self.population_number = pop_nr
        self.iterations = iterations_nr
        self.deposits = deposits
        self.capacity = capacity
        self.labels = None
        self.clusterMatrix = None
        self.solutions = self.newMultipleSolutions(self.population_number)
        self.history = []
        self.printEvery = 100

    def addPopulation(self, array):
        self.solutions.extend(array)
        self.population_number = len(self.solutions)


    def regenerate(self, percentage):
        mergeSort(self.solutions)
        # print "REGENERATED"
        new_solutions = self.generateMultipleSolutions(int(len(self.solutions)*percentage/100))
        # print "regenerated solutions:", len(new_solutions)
        for i in range(0, len(new_solutions)):
            self.solutions[-i-1] = new_solutions[i]

    def printBestSolution(self, top_solutions):
        global MUTATION_RATE

        best = self.solutions[:top_solutions]
        avg = sum(x.fitness for x in self.solutions)/len(self.solutions)

        mergeSort(best)

        for sol in self.solutions[top_solutions:]:
            for i in range(0, top_solutions):
                if best[i].fitness > sol.fitness :
                    best.insert( i, sol )
                    break

        self.history.append(best[0].fitness)
        top = map(lambda x: str(x.fitness), best[:top_solutions])

        CURSOR_UP_ONE = '\x1b[1A'
        ERASE_LINE = '\x1b[2K'
        if len(self.history) != 1:
            print CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE
        print "top self.solutions: " + " ".join(top) + "    avg: " + str(avg)

        length = 1.5
        progress = (100* len(self.history)*100/self.iterations)

        print "Progress[" + "#" *int(progress/length) + " "*int((100-progress)/length) + "]"

        best[0].filePrint()


    def generateSolutionNeighbours(self):
        sol = []
        temp_deposits = self.deposits[:]
        origin  = temp_deposits[0]
        del temp_deposits[0]

        # current = getBestNeighbour(origin, temp_deposits)
        current = temp_deposits[random.randint(0, len(temp_deposits)-1)]
        temp_deposits.remove(current)
        sol.append(current)


        while len(temp_deposits) > 0 :
            current = getBestNeighbour(current, temp_deposits)
            sol.append(current)
            temp_deposits.remove(current)
            # print current.demand

        # print len(sol)
        p = Permutation(origin, sol, self.capacity)
        return p

    def newMultipleSolutions(self, nr_solutions):
        solutions   = []

        # get labels of clusters for each deposit
        if self.labels is None :
            self.labels = get_labels(self.deposits)

        if self.clusterMatrix is None :
            self.clusterMatrix = get_clusters_matrix(self.deposits, self.labels)

        while len(solutions) < nr_solutions :
            s = self.generateSolutionNeighbours()
            # print s.fitness
            solutions.append(s)

        # print len(solutions)
        return solutions


    def generateMultipleSolutions(self, nr_solutions):
        global PROCESSORS

        if nr_solutions < PROCESSORS :
            PROCESSORS = nr_solutions

        # start with no solution
        solutions   = []

        # get labels of clusters for each deposit
        if self.labels is None :
            self.labels = get_labels(self.deposits)

        if self.clusterMatrix is None :
            self.clusterMatrix = get_clusters_matrix(self.deposits, self.labels)

        manager = Manager()
        solution = manager.dict()
        randomSolution = []

        for i in range(0,PROCESSORS):
            randomSolution.append(0)

        while len(solutions) < nr_solutions :
            if nr_solutions - len(solutions) < PROCESSORS :
                PROCESSORS =  nr_solutions - len(solutions)

            for pid in range(0, PROCESSORS):
                temp_deposits       = self.deposits[:]
                randomSolution[pid]   = Process(target = generateRandomSolution, args=(temp_deposits, self.capacity, solution, pid, self.labels, self.clusterMatrix) )
                randomSolution[pid].start()
            for pid in range(0,PROCESSORS):
                randomSolution[pid].join()
                solutions.append(solution[pid])

        return solutions

    def nextGeneration(self):
        self.solutions = crossParentsNew(self.solutions, self.clusterMatrix)

    def hillclimber(self, iterations):
        self.iterations = iterations
        self.history    = []
        for i in range(0, self.iterations):
            crossOver(self.solutions, self.clusterMatrix)
            if i % self.printEvery == 0:
                self.printBestSolution(3)

    def genetics(self, iterations):
        self.iterations = iterations
        self.history = []
        for i in range(0, self.iterations):
            self.nextGeneration()
            if i % self.printEvery == 0:
                self.printBestSolution(3)

    def both(self, iterations):
        self.iterations = iterations
        self.history = []

        for i in range(0, self.iterations):
            if i%2 == 0 :
                self.nextGeneration()
            else:
                crossOver(self.solutions, self.clusterMatrix)
            if i % self.printEvery == 0:
                self.printBestSolution(3)


def findPath(deposits, capacity):
    global MUTATION_RATE
    history = []
    solutions = []


    gen = []
    gen_combo = Generation(10, 10, deposits, capacity)
    # gen_combo.printBestSolution(1)
    for i in range(0, 4):
        gen.append(Generation(10, 10, deposits, capacity))
        MUTATION_RATE = 0.4
        gen[i].hillclimber(2000)
        gen[i].both(1000)
        gen[i].genetics(1000)
        MUTATION_RATE = 0.8
        gen[i].genetics(1000)
        solutions =  gen[i].solutions
        mergeSort(solutions)
        gen_combo.addPopulation(solutions[:10])
    # #
    MUTATION_RATE = 0.2
    # print len(gen_combo.solutions)
    gen_combo.both(5000)
    MUTATION_RATE = 0.8
    gen_combo.genetics(10000)
    # gen_combo.genetics(3000)


    plt.plot(range(len(gen_combo.history)), gen_combo.history)
    plt.show()

    return solutions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Input file with data not given")

    inputFile   = open(sys.argv[1])

    dimension   = int(inputFile.readline().split(":")[1])
    capacity    = int(inputFile.readline().split(":")[1])

    deposits    = []
    coords      = []

    # read coordinates
    line = inputFile.readline().strip("\n")
    while( line != "DEMAND_SECTION" ):
        if(line == "NODE_COORD_SECTION" ):
            line = inputFile.readline().strip("\n")
            continue
        tokens = map(int, line.split(" "))
        coords.append(Coordinates(tokens[1], tokens[2]))
        line = inputFile.readline().strip("\n")
    print "Read " + str(len(coords)) +  " coordinates"

    # read demand section
    for x in range(1, len(coords)+1):
        line = map(int, inputFile.readline().strip("\n").split(" "))
        deposits.append(Deposit(coords[line[0]-1], line[1], x-1))

    print "Read " + str(len(deposits)) +  " coordinates"

    sol1 = findPath(deposits, capacity)
