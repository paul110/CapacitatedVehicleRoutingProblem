from coordinates import Coordinates

class Cluster :
    def __init__(self, label, positions):
        self.positions = positions
        self.label = label
        self.centre = Coordinates(0,0)
        self.getCentre()

    def getCentre(self):
        for n in self.positions:
            self.centre.x += n.x
            self.centre.y += n.y
        self.centre.x = self.centre.x/len(self.positions)
        self.centre.y = self.centre.y/len(self.positions)
