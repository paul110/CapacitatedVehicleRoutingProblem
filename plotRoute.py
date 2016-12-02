import matplotlib.pyplot as plt
import random

def parsefile(filename):
    f = open(filename, 'r')
    coords = {}
    parselines = 0
    coordsparsed = 0
    for line in f:
        if "NODE" in line:
            parselines = 1
            continue
        elif parselines == 1:
            if "DEMAND" in line:
                parselines = 2
            else:
                temp = line.split(' ')
                coords[int(temp[0])] = (int(temp[1].rstrip()), int(temp[2].rstrip()))
        elif parselines == 0:
            continue
        else: break
    f.close()
    return coords

def parsesolution(file):
    f = open(file, 'r')
    route = []
    lineno = 0
    for line in f:
        if lineno == 4:
            temp = line.split('->')
            route.append(temp)
        else:
            lineno += 1
    return route

def get_houses(coords):
    x,y,depotx,depoty = [],[],0,0
    for key in coords:
        if key == 1:
            depotx = coords[key][0]
            depoty = coords[key][1]
        else:
            x.append(coords[key][0])
            y.append(coords[key][1])
    return depotx, depoty, x, y

def nodes_to_coords(coords, routes):
    routex, routey, temp, temp2 = [], [], [], []
    for route in routes:
        for node in route:
            temp.append(coords[int(node)][0])
            temp2.append(coords[int(node)][1])
        routex.append(temp)
        routey.append(temp2)
        temp, temp2 = [], []
    return routex,routey

def plot_data(depotx, depoty, x, y, routex, routey, colours, route_no, choice):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.axis([-100,100,-100,100])
    ax1.scatter(depotx, depoty, c='r')
    ax1.scatter(x,y, s=5, c='b')
    if choice == 1:
        ax1.plot(routex[route_no],routey[route_no],lw=0.5,c='b')
    elif choice == 0:
        for i in range(len(routex)):
            x = random.randint(0, 6)
            ax1.plot(routex[i],routey[i], lw=0.5, c=colours[x])
    elif choice == 2:
        for i in range(int(route_no[0]), int(route_no[1]) + 1):
            x = random.randint(0, 6)
            ax1.plot(routex[i],routey[i], lw=0.5, c=colours[x])
    else:
        for i in range(len(route_no)):
            x = random.randint(0, 6)
            ax1.plot(routex[int(route_no[i])],routey[int(route_no[i])],lw=0.5,c=colours[x])
    plt.show()

if __name__ == "__main__":
    route_no = -1
    plot = 1
    choice = int(raw_input("How many Routes? (0=All 1=Specify One 2=Specify Range 3=Specify N) "))
    if choice == 1:
        route_no = int(raw_input("Specify which route: "))
    elif choice == 2:
        route_no = raw_input("Specify which two routes separated by a space: ").split(" ")
    elif choice == 3:
        route_no = raw_input("Specify which routes separated by a space: ").split(" ")
    coords = parsefile("fruitybun250.vrp")
    routes = parsesolution("5810.txt")
    colours = 'bgrcmyk'
    depotx, depoty, x, y = get_houses(coords)
    routex, routey = nodes_to_coords(coords, routes)
    plot_data(depotx, depoty, x, y, routex, routey, colours, route_no, choice)
