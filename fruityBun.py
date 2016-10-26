import sys, random, math
from collections import namedtuple

NR_PATHS = 100


Coordinates = namedtuple("Coordinates", "x, y")

class Deposit:
    def __init__(self, position, demand, number):
        self.position   = position
        self.demand     = demand
        self.number     = number

class Path:
    def compute_fitness(self):
        distance = 0
        for x in range(1, len(self.route)):
            distance += dist(self.route[x-1].position, self.route[x].position)
        return distance

    def __init__(self, elements_order):
        self.route = elements_order
        self.fitness = self.compute_fitness()


def dist(location1, location2):
    return math.sqrt( pow(location1.x - location2.x, 2) + pow(location1.y - location2.y, 2) )

def generateMultiplePaths(deposits):
    global NR_PATHS
    paths = []
    for x in range(0, NR_PATHS):
        temp_deposits = deposits[:]
        paths.append( Path( generateRandomPath( temp_deposits ) ) )
        print paths[x].fitness
    return paths

def generateRandomPath(deposits):
    route = []
    route.append(deposits[0])
    deposits.remove(deposits[0])
    while(len(deposits) > 0):
        index = random.randint(0, len(deposits)-1)
        route.append(deposits[index])
        deposits.remove(deposits[index])
    return route;


def findPath(deposits):
    paths = generateMultiplePaths(deposits)


    return paths

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
        # print tokens[0]
    print "Read " + str(len(coords)) +  " coordinates"

    # read demand section
    for x in range(1, len(coords)+1):
        line = map(int, inputFile.readline().strip("\n").split(" "))
        deposits.append(Deposit(coords[line[0]-1], line[1], x-1))
    print "Read " + str(len(deposits)) +  " coordinates"

    path = findPath(deposits)
