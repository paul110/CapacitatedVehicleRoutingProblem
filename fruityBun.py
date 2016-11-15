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


TOP_SOLUTIONS   = 3
NR_SOLUTIONS    = 25
MUTATION_RATIO  = 201
ITERATIONS      = 10000
TRUCKS          = 25
PROCESSORS      = 4
CHANGES         = 2
THREADS         = 100
REPLACE         = 50

Coordinates = namedtuple("Coordinates", "x, y")
CrossPair   = namedtuple("CrossPair", "sol1, sol2")

class Deposit:
    def __init__(self, position, demand, number):
        self.position   = position
        self.demand     = demand
        self.number     = number
        self.label      = 0

# always returns the client from which the route
def findCrossOverPoint(solution):
    attempts = 1
    bestDist = None
    while attempts > 0:
        attempts -= 1
        index = random.randint(0, len(solution.destinations)-2)

        distance = dist(solution.destinations[index], solution.destinations[index+1])
        if (bestDist is None) or (distance < bestDist) :
            bestDist  = distance
            bestIndex = index

    client1 = solution.destinations[bestIndex]
    client2 = solution.destinations[bestIndex+1]

    return client1, client2

def combineSolutions(s1, s2):
    global CHANGES
    scopy1 = Permutation(s1.deposit, s1.destinations[:], s1.truck_capacity)
    scopy2 = Permutation(s2.deposit, s2.destinations[:], s2.truck_capacity)

    for _ in range(CHANGES):
        s1_client_1, s1_client_2  = findCrossOverPoint(s1)
        scopy2.linkDestinatios(s1_client_1, s1_client_2)

    for _ in range(CHANGES):
        s2_client_1, s2_client_2  = findCrossOverPoint(s2)
        scopy1.linkDestinatios(s2_client_1, s2_client_2)

    return scopy1,scopy2

def combineSolutions2(s1, s2):
    c1 = random.randint(0, 23)
    c2 = random.randint(0, 23)

    scopy1 = Permutation(s1.deposit, s1.destinations[:], s1.truck_capacity)
    scopy2 = Permutation(s2.deposit, s2.destinations[:], s2.truck_capacity)

    scopy1.swapClusters(c1, c2)
    scopy2.swapClusters(c1, c2)

    # print s1.fitness, scopy1.fitness, s2.fitness, scopy2.fitness

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
def crossOver(solutions):
    pairs   = []
    results = []
    while(len(solutions) > 1):
        first = solutions[random.randint(0, len(solutions)-1)]
        solutions.remove(first)

        second = solutions[random.randint(0, len(solutions)-1)]
        solutions.remove(second)

        pairs.append(CrossPair(first, second))

    for i in range(0, len(pairs)):
        first, second = combineSolutions2(pairs[i].sol1, pairs[i].sol2)
        first = chooseSolution(first, pairs[i].sol1)
        second = chooseSolution(second, pairs[i].sol2)
        first1, second1 = combineSolutions(first, second)
        first = chooseSolution(first, first1)
        second = chooseSolution(second, second1)
        solutions.extend([first, second])

def mutate(sol):
    index1 = random.randint(0, len(sol.destinations)-1)
    while True:
        index2 = random.randint(0, len(sol.destinations)-1)
        if index2 != index1 :
            break
    sol.swapNodes(index1, index2)


def mutations(solutions) :
    print "---- MUTATION ----"
    mutations = 0
    while mutations < len(solutions)/MUTATION_RATIO -1 :
        mutations +=1
        index = random.randint(0, len(solutions)-1)
        mutate(solutions[index])



class Permutation:
    def __init__(self, deposit, destinations, capacity):
        self.deposit        = deposit
        self.destinations    = destinations
        self.truck_capacity = capacity
        self.fitness        = 0
        self.up_to_date     = False

        self.update()

    def update(self):
        self.computeFitness()

    def swapNodes(self, index1, index2):
        self.up_to_date = False

        aux = self.destinations[index1]
        self.destinations[index1] = self.destinations[index2]
        self.destinations[index2] = aux

        self.update()

    def swapClusters(self, c1, c2):
        self.up_to_date = False
        c1_start, c1_end = self.findCluster(c1)
        c2_start, c2_end = self.findCluster(c2)

        while c1_start < c1_end and c2_start < c2_end:
            aux = self.destinations[c1_start]
            self.destinations[c1_start] = self.destinations[c2_start]
            self.destinations[c2_start] = aux
            c1_start += 1
            c2_start += 1

        while c1_start < c1_end:
            self.destinations.insert(c2_start, self.destinations[c1_start])
            if c1_start > c2_start :
                c1_start += 1
                c2_start += 1
            else :
                c1_end -= 1

            del self.destinations[c1_start]

        while c2_start < c2_end:
            self.destinations.insert(c1_start, self.destinations[c2_start])
            if c2_start > c1_start :
                c2_start += 1
                c1_start += 1
            else :
                c2_end -= 1

            del self.destinations[c2_start]

        self.update()


    def findCluster(self, cluster):
        start = None
        end =   None
        for i in range(0, len(self.destinations)):
            if self.destinations[i].label == cluster :
                start = i
                end = i
                break
        i = start
        # print i, len(self.destinations), cluster
        while i < len(self.destinations) and self.destinations[i].label == cluster:
            end += 1
            i   += 1

        return start, end

    # !!! does not trigger update automatically
    def insertElements(self, index, elements):
        self.up_to_date = False

        removed = 0
        for d in self.destinations :
            for e in elements :
                if e.number == d.number :
                    self.destinations.remove(d)
                    removed += 1
                    break
            if removed == len(elements) :
                break

        for i in range(0, len(elements)):
            self.destinations.insert( index+i, elements[i] )

        self.update()

    def linkDestinatios(self, d1, d2):
        self.up_to_date = False

        d1_index = 0
        d2_index = 0


        for i in range(0, len(self.destinations)):
            # print i, len(self.destinations)
            if d2.number == self.destinations[i].number :
                d2_index = i
            if d1.number == self.destinations[i].number :
                d1_index = i

        del self.destinations[d2_index]
        self.destinations.insert(d1_index+1, d2)

        self.update()

    def computeFitness(self):
        capacity        = self.truck_capacity - self.destinations[0].demand
        self.fitness    = dist(self.deposit, self.destinations[0])

        for i in range(0, len(self.destinations)-1):
            nextDest = self.destinations[i+1]
            previousDest = self.destinations[i]

            if nextDest.demand > capacity:
                # go to deposit first and then to next destination
                self.fitness += dist(previousDest, self.deposit)

                # reset capacity
                capacity = self.truck_capacity

                # update preious destination
                previousDest = self.deposit

            # update capacity
            capacity -= nextDest.demand

            # travel to next destination
            self.fitness += dist(previousDest, nextDest)

        # go back to deposit
        self.fitness += dist(self.destinations[-1], self.deposit)

    def filePrint(self) :
        f = open('best-solution.txt', 'w')
        f.write("login pp13003 68443\n")
        f.write("name Paul Pintilie\n")
        f.write("algorithm Genetic Algorithm with specialized crossover and mutation\n")
        f.write("cost "+ str(self.fitness) + "\n")
        string = "1->" +str( self.destinations[0].number + 1 )
        capacity = self.truck_capacity - self.destinations[0].demand
        for i in range(0, len(self.destinations)-1):
            nextDest = self.destinations[i+1]
            previousDest = self.destinations[i]

            if nextDest.demand > capacity:
                # go to deposit first and then to next destination
                string += "->1\n1"
                # reset capacity
                capacity = self.truck_capacity

                # update preious destination
                previousDest = self.deposit

            # update capacity
            capacity -= nextDest.demand

            # travel to next destination
            string += "->" + str(nextDest.number+1)

        # go back to deposit
        # self.fitness += dist(self.destinations[-1], self.deposit)
        string += "->" + str(self.deposit.number+1) + "\n"
        f.write(string)

class Truck:
    def __init__(self, capacity, deposit):
        self.capacity = capacity
        self.traveled = 0
        self.route = [deposit, deposit]

    def addDestination(self, destination):
        self.route.insert(len(self.route)-1, destination)

        self.traveled += dist(self.route[-1], self.route[-2])
        self.traveled += dist(self.route[-2], self.route[-3])
        self.traveled -= dist(self.route[-1], self.route[-3])

        self.capacity -= destination.demand

    def removeNode(self):
        index = random.randint(1, len(self.route)-2)
        node = self.route[index]
        self.route.remove(self.route[index])

        self.capacity += node.demand

        self.traveled -= dist(self.route[-1], node)
        self.traveled -= dist(self.route[-2], node)
        self.traveled += dist(self.route[-1], self.route[-2])

        return node

    def addNode(self, index, node):
        self.route.insert(index, node)

        self.traveled += dist(node, self.route[index+1])
        self.traveled += dist(node, self.route[index-1])
        self.traveled -= dist(self.route[index+1], self.route[index-1])

        self.capacity -= node.demand

    def changeDestination(self, index, newNode):
        # resotre capacity spent on previous destination
        old_capacity = self.route[index].demand
        self.capacity += old_capacity

        old_distance        = dist(self.route[index], self.route[index+1]) + dist(self.route[index], self.route[index-1])
        self.route[index]   = newNode
        self.capacity -= newNode.demand
        new_distance        = dist(self.route[index], self.route[index+1]) + dist(self.route[index], self.route[index-1])

        self.traveled += new_distance - old_distance

def mergeSort(solutions):
    if len(solutions) > 1 :
        mid         = len(solutions)//2
        lefthalf    = solutions[:mid]
        righthalf   = solutions[mid:]
        mergeSort(lefthalf)
        mergeSort(righthalf)

        i = 0
        j = 0
        k = 0
        while i<len(lefthalf) and j < len(righthalf):
            if(lefthalf[i].fitness < righthalf[j].fitness ):
                solutions[k] = lefthalf[i]
                i+=1
            else :
                solutions[k] = righthalf[j]
                j+=1
            k+=1

        while i < len(lefthalf):
            solutions[k] = lefthalf[i]
            i+=1
            k+=1

        while j < len(righthalf):
            solutions[k] = righthalf[j]
            j+=1
            k+=1

def dist(location1, location2):
    return math.sqrt( pow(location1.position.x - location2.position.x, 2) + pow(location1.position.y - location2.position.y, 2) )

def get_clusters(deposits):
    coords = map(lambda dep: [float(dep.position.x), float(dep.position.y)] , deposits)
    y = np.array(coords)
    codebook, _ = kmeans(y, 25)  # three clusters
    cluster_indeces, _ = vq(y, codebook)

    xpoints = [str(d.position.x) for d in deposits]
    ypoints = [str(d.position.y) for d in deposits]

    for i in range(0, len(cluster_indeces)):
        deposits[i].label = cluster_indeces[i]

    return cluster_indeces

def generateMultipleSolutions(deposits, capacity, nr_solutions, labels=None):
    global PROCESSORS

    if nr_solutions < PROCESSORS :
        PROCESSORS = nr_solutions

    # start with no solution
    solutions   = []


    # get labels of clusters for each deposit
    if labels is None :
        labels = get_clusters(deposits)

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
            randomSolution[pid]   = Process(target = generateRandomSolution, args=(temp_deposits, capacity, solution, pid, labels) )
            randomSolution[pid].start()
        for pid in range(0,PROCESSORS):
            randomSolution[pid].join()
            solutions.append(solution[pid])

    return solutions, labels

def generateRandomSolution(deposits, capacity, solution, sol_index, depo_labels = []):
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

        while len(failed) > 0 :
            attempt = False
            for truck in trucks:
                if truck.capacity >= failed[0].demand :
                    truck.addDestination(failed[0])
                    failed.remove(failed[0])
                    attempt = True
                    break
            if not attempt :
                trucks.append(Truck(capacity, startDeposit))
                trucks[-1].addDestination(failed[0])
                failed.remove(failed[0])

    while trucks :
        index = random.randint(0, len(trucks)-1)
        destinations.extend(trucks[index].route[1:-1])
        del trucks[index]

    solution[sol_index] = Permutation(startDeposit, destinations, capacity)

def printBestSolution(solutions, history):
    global TOP_SOLUTIONS
    best = []
    if len(solutions) < TOP_SOLUTIONS:
        TOP_SOLUTIONS = len(solutions)

    # update fitness
    for sol in solutions:
        sol.update()

    for i in range(0, TOP_SOLUTIONS):
        best.append(solutions[i])

    mergeSort(best)

    avg = 0
    for sol in solutions:
        avg += sol.fitness
    avg = avg / len(solutions)

    for sol in solutions[TOP_SOLUTIONS:]:
        for i in range(0, TOP_SOLUTIONS):
            if best[i].fitness > sol.fitness :
                best.insert( i, sol )
                break

    history.append(best[0].fitness)

    top = map(lambda x: str(x.fitness), best[:TOP_SOLUTIONS])
    print "top solutions: " + " ".join(top) + "    avg: " + str(avg)

    best[0].filePrint()

def plotPoints(sol):

    xpoints = [str(dest.position.x) for dest in sol.trucks[0].route]
    ypoints = [str(dest.position.y) for dest in sol.trucks[0].route]

    print xpoints[0]
    plt.scatter(xpoints, ypoints)
    plt.show()

def regenerate(solutions, deposits, capacity, labels):
    mergeSort(solutions)

    new_solutions, labels = generateMultipleSolutions(deposits, capacity, int(len(solutions)*REPLACE/100), labels)
    print "regenerated solutions:", len(new_solutions)
    for i in range(0, len(new_solutions)):
        solutions[-i-1] = new_solutions[i]
    # printBestSolution(new_solutions)

def findPath(deposits, capacity):
    global ITERATIONS
    history = []

    # generate start population
    solutions, labels = generateMultipleSolutions(deposits, capacity, NR_SOLUTIONS)

    # test(solutions)
    printBestSolution(solutions, history)
    for i in range(0, ITERATIONS):
        crossOver(solutions)

        if i%100 == 0  :
            regenerate(solutions, deposits, capacity, labels)
            mutations(solutions)
        if i%10 == 0 :
            printBestSolution(solutions, history)

    # solutions[0].swapClusters(1,2)
    plt.plot(range(len(history)), history)
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

    path = findPath(deposits, capacity)
