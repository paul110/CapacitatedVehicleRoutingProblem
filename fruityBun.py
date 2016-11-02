import sys, random, math
from collections import namedtuple
import matplotlib.pyplot as plt
import scipy.cluster.vq as clustering
import numpy as np
from scipy.cluster.vq import kmeans, vq
import threading
import gevent
from gevent import Greenlet
from multiprocessing import Process, Manager

TOP_SOLUTIONS   = 4
NR_SOLUTIONS    = 40
ITERATIONS      = 10000
TRUCKS          = 25
PROCESSORS      = 4

Coordinates = namedtuple("Coordinates", "x, y")
CrossPair   = namedtuple("CrossPair", "sol1, sol2")

class Deposit:
    def __init__(self, position, demand, number):
        self.position   = position
        self.demand     = demand
        self.number     = number
        self.label      = 0

def findBestPair(truck):
    start = truck.route[1]
    end = truck.route[2]
    best_dist = dist(truck.route[1], truck.route[2])
    for i in range(3, len(truck.route)-2):
        if dist(truck.route[i-1], truck.route[i]) < best_dist:

            best_dist   = dist(truck.route[i-1], truck.route[i])
            start       = truck.route[i-1]
            end         = truck.route[i]
    return start, end

# always returns the client from which the route
def findCrossOverPoint(solution):
    while True:
        truck_index = random.randint(0, len(solution.trucks)-1)
        truck       = solution.trucks[truck_index]
        if len(truck.route) > 3:
            break

    client1, client2 = findBestPair(truck)
    if client1.number == 0 or client2.number == 0 :
        print "TATA", client1.number, client2.number

    return client1, client2

# find position of client in a solution ( returns truck_inedx, path_index )
def findPosition(client, solution):
    trucks = solution.trucks
    for i in range(0, len(trucks)):
        for j in range(0, len(trucks[i].route)):
            if trucks[i].route[j].number == client.number :
                return i, j
    return 0, 0

def combineSolutions(s1, s2):
    s1_client_1, s1_client_2  = findCrossOverPoint(s1)
    s2_client_1, s2_client_2  = findCrossOverPoint(s2)

    scopy1 = Solution(s1.trucks)
    scopy2 = Solution(s2.trucks)
    scopy1.addLink(s2_client_1, s2_client_2)
    scopy2.addLink(s1_client_1, s1_client_2)
    return scopy1,scopy2

def chooseSolution(sol1, sol2):
    if sol1.fitness > sol2.fitness :
        return sol2
    else:
        return sol1

# crossover between 2 solution
def crossOver(solutions):
    # print "--------CROSSOVER--------"
    pairs = []
    while(len(solutions) > 1):
        first = solutions[0]
        solutions.remove(first)

        second = solutions[-1]
        solutions.remove(second)

        pairs.append(CrossPair(first, second))

    # can be paralelised easily
    for pair in pairs:
        first, second = combineSolutions(pair.sol1, pair.sol2)
        # print first.fitness, second.fitness, pair.sol1.fitness, pair.sol2.fitness
        solutions.append(chooseSolution(first, pair.sol1))
        solutions.append(chooseSolution(second, pair.sol2))

# def crossOver2(solutions):

class Solution:
    def __init__(self, trucks):
        self.trucks = trucks
        self.capacity_left = 0
        self.traveled = 0
        for truck in trucks:
            self.capacity_left += truck.capacity
            self.traveled += truck.traveled
        self.fitness = self.traveled

    # path_index = index of node which will stay there
    # path_index_dest = index of node which will be changed
    # newClient = node which will be placed at the path_index_dest
    def changeNode(self, truck_index, path_index, dest_path_index, newClient, linkClient):

        # find the position of the node which will be placed at path_index_dest already has in the solution
        replace_truck_index, replace_path_index = findPosition(newClient,  self)
        # finePrint(self)

        # if replace_path_index == 0 :
        #     print "NU BINE", newClient.number

        # remember the Client which is being taken out of the route
        backupClient    = self.trucks[truck_index].route[dest_path_index]

        # if(newClient.demand - backupClient.demand) > self.trucks[truck_index].capacity:
        #     print "WTF HAPPPENED"
        #     return
        # replace client with the new one
        self.trucks[truck_index].route[dest_path_index] = newClient

        # modify the solution distance
        self.traveled += dist(newClient, linkClient) - dist(backupClient, linkClient)
        if dest_path_index - path_index > 0:
            self.traveled += dist(newClient, self.trucks[truck_index].route[dest_path_index+1])
            self.traveled -= dist(backupClient, self.trucks[truck_index].route[dest_path_index+1])
        else :
            self.traveled += dist(newClient, self.trucks[truck_index].route[dest_path_index-1])
            self.traveled -= dist(backupClient, self.trucks[truck_index].route[dest_path_index-1])

        # replace duplicate node with missing node ( the back up )

        self.trucks[replace_truck_index].route[replace_path_index] = backupClient

        # add new distances
        self.traveled += dist(backupClient, self.trucks[replace_truck_index].route[replace_path_index - 1])
        self.traveled -= dist(newClient, self.trucks[replace_truck_index].route[replace_path_index - 1])
        self.traveled += dist(backupClient, self.trucks[replace_truck_index].route[replace_path_index + 1])
        self.traveled -= dist(newClient, self.trucks[replace_truck_index].route[replace_path_index + 1])

        self.fitness = self.traveled

    def addLink(self, client1, client2):
        # find position of client1 node
        truck_index, p_index = findPosition(client1, self)

        route = self.trucks[truck_index].route

        # if change can be made in eiher direction
        if (p_index+1 < len(route)-1) and (p_index-1 > 0) :
            leftDistance    = 0
            rightDistance   = 0

            leftDistance = dist(route[p_index], route[p_index-1])
            rightDistance = dist(route[p_index], route[p_index+1])

            if leftDistance > rightDistance :
                self.changeNode(truck_index, p_index, p_index-1, client2, client1)
            else:
                self.changeNode(truck_index, p_index, p_index+1, client2, client1)
        else:
            if p_index+1 < len(route)-1:
                self.changeNode(truck_index, p_index, p_index+1, client2, client1)
            else :
                if p_index-1 > 0 :
                    self.changeNode(truck_index, p_index, p_index-1, client2, client1)

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

class Path:
    def compute_fitness(self):
        distance = 0
        for x in range(1, len(self.route)):
            distance += dist(self.route[x-1], self.route[x])
        return distance

    def __init__(self, elements_order):
        self.route = elements_order
        self.fitness = self.compute_fitness()

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

    fig, ax = plt.subplots()
    ax.scatter(xpoints, ypoints)

    for i, ind in enumerate(cluster_indeces):
        ax.annotate(ind, (xpoints[i],ypoints[i]))

    # plt.show()
    for i in range(0, len(cluster_indeces)):
        deposits[i].label = cluster_indeces[i]

    return cluster_indeces

class generateRandSol(threading.Thread):
    def __init__(self, deposits, capacity, labels):
        threading.Thread.__init__(self)
        self.deposits   = deposits
        self.capacity   = capacity
        self.labels     = labels
    def run(self):
        self.solution = generateRandomSolution(self.deposits, self.capacity, self.labels)

    def join(self):
        return self.solution

def generateMultipleSolutions(deposits, capacity):
    global NR_SOLUTIONS
    global PROCESSORS

    if NR_SOLUTIONS < PROCESSORS :
        PROCESSORS = NR_SOLUTIONS

    # start with no solution
    solutions   = []

    # get labels of clusters for each deposit
    labels = get_clusters(deposits)

    manager = Manager()
    solution = manager.dict()
    # solution = []

    randomSolution = []
    for i in range(0,PROCESSORS):
        randomSolution.append(0)

    while len(solutions) < NR_SOLUTIONS :
        for pid in range(0, PROCESSORS):
            temp_deposits       = deposits[:]
            randomSolution[pid]   = Process(target = generateRandomSolution, args=(temp_deposits, capacity, solution, pid, labels) )
            randomSolution[pid].start()
        for pid in range(0,PROCESSORS):
            randomSolution[pid].join()
            solutions.append(solution[pid])

    return solutions

def generateRandomSolution(deposits, capacity, solution, sol_index, depo_labels = []):
    global TRUCKS
    trucks = []

    startDeposit = deposits[0];
    deposits.remove(deposits[0])

    # initialise all trucks
    for i in range(0, TRUCKS):
        trucks.append(Truck(capacity, startDeposit))

    if len(depo_labels) == 0 :
        while(len(deposits) > 0):
            allocated = False
            index = random.randint(0, len(deposits)-1)
            for truck in trucks:
                if truck.capacity >= deposits[index].demand :
                    truck.addDestination(deposits[index])
                    allocated = True
                    break

            if not allocated :
                trucks.append(Truck(capacity, startDeposit))
                trucks[-1].addDestination(deposits[index])

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

    solution[sol_index] = Solution(trucks)

def findFittestPath(paths):
    best = paths[0]
    for path in paths:
        if(path.fitness < best.fitness):
            best = path
    return best

def printSolutions(solutions):
    for s in solutions:
        print "traveled: %5.4f with %2d trucks and %4d capacity left" %(s.traveled, len(s.trucks), s.capacity_left)
        trucks = s.trucks

def finePrint(solution):
    f = open('best-solution.txt', 'w')
    f.write("login pp13003 21124\n")
    f.write("name Paul Pintilie\n")
    f.write("algorithm Genetic Algorithm with specialized crossover and mutation\n")
    f.write("cost "+ str(solution.fitness) + "\n")
    for truck in solution.trucks:
        a = map(lambda x: str(x.label+1), truck.route)
        f.write("->".join(a) +  "\n")

def printBestSolution(solutions):
    global TOP_SOLUTIONS
    best = []
    if len(solutions) < TOP_SOLUTIONS:
        TOP_SOLUTIONS = len(solutions)

    for i in range(0, TOP_SOLUTIONS):
        best.append(solutions[i])

    mergeSort(best)

    for sol in solutions[TOP_SOLUTIONS:]:
        for i in range(0, TOP_SOLUTIONS):
            if best[i].fitness > sol.fitness :
                best.insert( i, sol )
                break

    top = map(lambda x: str(x.fitness), best[:TOP_SOLUTIONS])
    print "top solutions: " + " ".join(top)
    # print "top solution: " + str(best.fitness )
    finePrint(best[0])
    # plotPoints(best)

def plotPoints(sol):

    xpoints = [str(dest.position.x) for dest in sol.trucks[0].route]
    ypoints = [str(dest.position.y) for dest in sol.trucks[0].route]

    print xpoints[0]
    plt.scatter(xpoints, ypoints)
    plt.show()

def findPath(deposits, capacity):
    global ITERATIONS
    # generate start population
    solutions = generateMultipleSolutions(deposits, capacity)
    # sor population based on fitness
    printSolutions(solutions)
    printBestSolution(solutions)
    for i in range(0, ITERATIONS):
        crossOver(solutions)
        if i%10 == 0  :
            printBestSolution(solutions)

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
