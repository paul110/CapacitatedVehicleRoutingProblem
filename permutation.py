from helperFunctions import *

class Permutation:
    def __init__(self, deposit, destinations, capacity):
        self.deposit        = deposit
        self.destinations    = destinations
        self.truck_capacity = capacity
        self.fitness        = 0
        self.up_to_date     = False
        self.trucks = []

        self.update()

    def update(self):
        self.computeFitness()

    def swapNodes(self, index1, index2):
        self.up_to_date = False

        aux = self.destinations[index1]
        self.destinations[index1] = self.destinations[index2]
        self.destinations[index2] = aux

        self.update()


    def findCrossOverPoint(self):

        attempts = 1
        bestDist = None
        while attempts > 0:
            attempts -= 1
            index = random.randint(0, len(self.destinations)-2)

            distance = dist(self.destinations[index], self.destinations[index+1])
            if (bestDist is None) or (distance < bestDist) :
                bestDist  = distance
                bestIndex = index

        client1 = self.destinations[bestIndex]
        client2 = self.destinations[bestIndex+1]

        return client1, client2

    def getClusterNeighbours(self, cluster, totalClusters):
        curr = 0
        neighbours = []
        i = 0
        while i < len(self.destinations):
            d = self.destinations[i]
            if d.label == cluster:
                neighbours.append(curr)
                i += 1
                while i < len(self.destinations) and d.label == cluster :
                    d = self.destinations[i]
                    i +=1

                if d.label == cluster:
                    break
                else:
                    neighbours.append(d.label)

                break
            curr = d.label
            i+=1
        if len(neighbours) < 1:
            print "empty"
            neighbours.append(random.randint(0,totalClusters))
        return neighbours

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
        bottomSearch = random.randint(0,1)
        # minimum = random.randint(1, 3)
        minimum = 1
        if bottomSearch :
            start = 0
            end = 1
            for i in range(0, len(self.destinations)):
                if self.destinations[i].label == cluster :
                    start = i
                    end = start
                    while end < len(self.destinations) and self.destinations[end].label == cluster and end - start < minimum:
                        end += 1
                    if end-start > minimum-1 or end >= len(self.destinations):
                        break

            # print i, len(self.destinations), cluster
            while end < len(self.destinations) and self.destinations[end].label == cluster:
                end += 1

        else:
            start = len(self.destinations)-1
            end = start + 1
            for i in range(len(self.destinations)-1, -1, -1):
                if self.destinations[i].label == cluster:
                    end = i+1
                    start = i
                    while start-1 >= 0 and self.destinations[start-1].label == cluster and end - start + 1 < minimum:
                        start -=1
                    if end - start > minimum - 1 or start == 0:
                        break
            while start -1 >= 0 and self.destinations[start-1].label == cluster:
                start -= 1

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
        if d2_index < d1_index :
            d1_index +=1

        self.destinations.insert(d1_index+1, d2)

        self.update()

    def invert(self, idx1, idx2):
        id1 = min(idx1, idx2)
        id2 = max(idx1, idx2)

        while id1 < id2 :
            aux = self.destinations[id1]
            self.destinations[id1] = self.destinations[id2]
            self.destinations[id2] = aux
            id1 += 1
            id2 -= 1
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

    def printOut(self, filePrint) :
        f = open('best-solution.txt', 'w')
        string = "login pp13003 68443\n"
        string += "name Paul Pintilie\n"
        string += "algorithm Genetic Algorithm with specialized crossover and mutation\n"
        string += "cost "+ str(self.fitness) + "\n"
        string += "1->" +str( self.destinations[0].number + 1 )
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
        if filePrint:
            f.write(string)
            return
        print string

    def plotDestinations(self):

        xpoints = [str(dest.position.x) for dest in sol.trucks[0].route]
        ypoints = [str(dest.position.y) for dest in sol.trucks[0].route]

        print xpoints[0]
        plt.scatter(xpoints, ypoints)
        plt.show()
