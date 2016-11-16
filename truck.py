from helperFunctions import *

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
