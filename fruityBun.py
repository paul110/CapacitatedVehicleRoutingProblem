import sys, random, math
from collections import namedtuple

NR_PATHS    = 20
ITERATIONS  = 100000

Coordinates = namedtuple("Coordinates", "x, y")
CrossPair   = namedtuple("CrossPair", "sol1, sol2")

class Deposit:
    def __init__(self, position, demand, number):
        self.position   = position
        self.demand     = demand
        self.number     = number

def findBestPair(truck):
    start = truck.route[1]
    end = truck.route[2]
    best_dist = dist(truck.route[0], truck.route[1])
    for i in range(3, len(truck.route)-1):
        if dist(truck.route[i-1], truck.route[i]) < best_dist:
            best_dist   = dist(truck.route[i-1], truck.route[i])
            start       = truck.route[i-1]
            end         = truck.route[i]
    return start, end

# always returns the client from which the route
def findCrossOverPoint(solution):
    truck_index = random.randint(0, len(solution.trucks)-1)
    truck       = solution.trucks[truck_index]

    client1, client2 = findBestPair(truck)
    return truck_index, client1, client2


def findPosition(client, solution):
    trucks = solution.trucks
    for i in range(0,len(trucks)):
        t = trucks[i]
        for j in range(0, len(t.route)):
            if t.route[j] == client:
                return i, j
    return 0, 0

def combineSolutions(s1, s2):
    s1_truck, s1_client_1, s1_client_2  = findCrossOverPoint(s1)
    s2_truck, s2_client_1, s2_client_2  = findCrossOverPoint(s2)

    s1.addLink(s2_client_1, s2_client_2)
    s2.addLink(s1_client_1, s1_client_2)
    return s1,s2

# crossover between 2 solution
def crossOver(solutions):
    pairs = []
    while(len(solutions) > 2):
        first = random.randint(0, len(solutions)-1)
        first = solutions[first]
        solutions.remove(first)

        second = random.randint(0,len(solutions)-1)
        second = solutions[second]
        solutions.remove(second)

        pairs.append(CrossPair(first, second))

    # can be paralelised easily
    for i in range(0,len(pairs)):
        first, second = combineSolutions(pairs[i].sol1, pairs[i].sol2)
        solutions.append(first)
        solutions.append(second)




class Solution:
    def __init__(self, trucks):
        self.trucks = trucks
        self.capacity_left = 0
        self.traveled = 0
        for x in range(0, len(trucks)):
            self.capacity_left += trucks[x].capacity
            self.traveled += trucks[x].traveled
        self.fitness = self.traveled

    # path_index = index of node which will stay there
    # path_index_dest = index of node which will be changed
    # newClient = node which will be placed at the path_index_dest
    def changeNode(self, truck_index, path_index, dest_path_index, newClient, linkClient):
        # find the position of the node which will be placed at path_index_dest already has in the solution
        replace_truck_index, replace_path_index = findPosition(newClient,  self)
        # print len(self.trucks[truck_index].route), path_index, dest_path_index
        # print len(self.trucks[replace_truck_index].route), replace_path_index
        # print self.traveled
        # remember the Client which is being taken out of the route
        backupClient    = self.trucks[truck_index].route[dest_path_index]

        # replace client with the new one
        self.trucks[truck_index].route[dest_path_index] = newClient
        # self.capacity_left -= newClient

        # modify the solution distance
        self.traveled += dist(newClient, linkClient) - dist(backupClient, linkClient)
        if dest_path_index - path_index > 0:
            # print "prima"
            self.traveled += dist(newClient, self.trucks[truck_index].route[dest_path_index+1])
            self.traveled -= dist(backupClient, self.trucks[truck_index].route[dest_path_index+1])
        else :
            # print "doua"
            self.traveled += dist(newClient, self.trucks[truck_index].route[dest_path_index-1])
            self.traveled -= dist(backupClient, self.trucks[truck_index].route[dest_path_index-1])


        # find the duplic   ate node
        # print self.trucks

        # replace duplicate node with missing node ( the back up )
        self.trucks[replace_truck_index].route[replace_path_index] = backupClient
        # add new distances

        self.traveled += dist(backupClient, self.trucks[replace_truck_index].route[replace_path_index - 1])
        self.traveled -= dist(newClient, self.trucks[replace_truck_index].route[replace_path_index - 1])
        self.traveled += dist(backupClient, self.trucks[replace_truck_index].route[replace_path_index + 1])
        self.traveled -= dist(newClient, self.trucks[replace_truck_index].route[replace_path_index + 1])

        self.fitness = self.traveled
        # print self.traveled
        # substract old distances
        # self.traveled -= ( dist(newClient, self.trucks[truck_index].route[path_index - 1]) + dist(newClient, self.trucks[truck_index].route[path_index + 1]) )


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

            if p_index-1 > 1 :
                self.changeNode(truck_index, p_index, p_index-1, client2, client1)

class Truck:
    def __init__(self, capacity, deposit):
        self.capacity = capacity
        self.traveled = 0
        self.route = [deposit]

    def addDestination(self, destination):
        self.traveled += dist(self.route[-1], destination)
        self.capacity -= destination.demand
        self.route.append(destination)

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

def generateMultipleSolutions(deposits, capacity):
    global NR_PATHS
    solutions = []
    for x in range(0, NR_PATHS):
        temp_deposits = deposits[:]
        solutions.append( generateRandomSolution( temp_deposits, capacity ) )
    return solutions

def generateRandomSolution(deposits, capacity):
    trucks = []
    startDeposit = deposits[0];
    deposits.remove(deposits[0])

    trucks.append(Truck(capacity, startDeposit))
    while(len(deposits) > 0):
        allocated = 0
        index = random.randint(0, len(deposits)-1)
        for x in range(0, len(trucks)):
            if trucks[x].capacity > deposits[index].demand :
                trucks[x].addDestination(deposits[index])
                allocated = 1
                break
        if not allocated :
            trucks.append(Truck(capacity, startDeposit))
            trucks[-1].addDestination(deposits[index])

        deposits.remove(deposits[index])

    for x in range(0, len(trucks)):
        trucks[x].addDestination(startDeposit)

    return Solution(trucks);

def findFittestPath(paths):
    best = paths[0]
    for x in range(1,len(paths)-1):
        if(paths[x].fitness < best.fitness):
            best = paths[x]
    return best

def printSolutions(solutions):
    for i in range(0, NR_PATHS):
        print "travelled: " + str(solutions[i].traveled) + "   capacity: " + str(solutions[i].capacity_left)
        trucks = solutions[i].trucks
        # for j in range(0, len(trucks)):
        #     start, end = findBestPair(trucks[j])
        #     print "truck:" +  str(j) +"   "  + str(start.number) + "--"+ str(end.number) + " = "+ str(dist(start,end))
def printBestSolution(solutions):
    best = solutions[0]
    for i in range(1, len(solutions)):
        if solutions[i].fitness < best.fitness :
            best = solutions[i]

    print "top solution: " + str(best.fitness )

def findPath(deposits, capacity):
    global ITERATIONS
    # generate start population
    solutions = generateMultipleSolutions(deposits, capacity)
    # sor population based on fitness
    printSolutions(solutions)
    for i in range(0, ITERATIONS):
        # first = random.randint(0, len(solutions)-1)
        # while True:
        #     second = random.randint(0,len(solutions)-1)
        #     if second != first:
        #         break
        crossOver(solutions)
        # crossOver(solutions[first], solutions[second])
        if i%100 == 0  :
            printBestSolution(solutions)
    print "--------------------"
    # mergeSort(solutions)
    # printSolutions(solutions)



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
