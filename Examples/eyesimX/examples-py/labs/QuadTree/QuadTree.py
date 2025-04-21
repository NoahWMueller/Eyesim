#!/usr/bin/env python
#Noah Mueller 23356892 24/04/2024
from eye import *
import math
import ctypes

# Globals ----------------------------------------------------------------------

nodes = [None] * 1000
occupiedSquares = [None] * 1000
freeSquareCount = 0 
occupiedSquareCount = 0
node_id = 1
occ_id = 1

WORLD_SIZE = 3990
IMAGE_SIZE = 128
IMAGE_FILE = 'corner.pbm'
GOAL_X = 3800
GOAL_Y = 200
START_X = 200
START_Y = 3800
BUFFER = 50

# classes ----------------------------------------------------------------------

class occupiedSquare:
    def __init__(self, id: int, x: int, y: int, size: int):
        self.id = id
        buffer = 8
        self.x = x-int(buffer/2)
        self.y = y-int(buffer/2)
        self.size = size + int(buffer)

class Node:
    def __init__(self, id: int, x: int, y: int):
        self.id = id
        self.x = x
        self.y = y
        self.neighbours = []
    def addNeighbour(self, neighbour):
        self.neighbours.append(neighbour)

class Edge:
    def __init__(self, neighbour: Node, distance: int):
        self.neighbour = neighbour
        self.distance = distance

# functions ----------------------------------------------------------------------

def quadTree(x, y, size, img):
    global freeSquareCount
    global occupiedSquareCount
    global node_id
    global occ_id
    global nodes
    global occupiedSquares

    allFree = True
    allOcc = True
    # start_node = Node(0, int(200/120), int(200/120))
    # nodes[0] = start_node

    for i in range(x, x+size):
        for j in range(y, y+size):
            if (img[IMAGE_SIZE*j+i]): 
                # at least 1 occ.
                allFree=False
            else:
                # at least 1 free
                allOcc=False
    
    if (allFree):
        # draw the square
        LCDArea(x + 1 +BUFFER,y + 1+BUFFER, x + size - 1+BUFFER, y + size -  1+BUFFER, 0x00FF00, 0)
        if size > 4:
            freeSquareCount += 1
            node = nodes[node_id-1]
            if node is None:
                node = Node(node_id, x+int(size/2), y+int(size/2))
                nodes[node_id-1] = node
            node_id+=1

    if (allOcc):
        # draw the square
        LCDArea(x + 1 +BUFFER,y + 1+BUFFER, x + size - 1+BUFFER, y + size -  1+BUFFER, 0xFF0000, 0)
        occupiedSquareCount+=1     
        square = occupiedSquares[occ_id-1]
        if square is None:
            square = occupiedSquare(occ_id,x,y,size)  
            occupiedSquares[occ_id-1] = square
        occ_id+=1

    if (not allOcc and not allFree and (size>1)):
        s2 = round(size/2)
        quadTree(x, y, s2, img)
        quadTree(x+s2, y, s2, img)
        quadTree(x, y+s2, s2, img)
        quadTree(x+s2, y+s2, s2, img)

def read_image_file(filename):
    """
    :param filename: The name of the PBM image file to read
    :return: The image as an array of integers, the width, the height
    """

    f = open(filename, 'r')
    f.readline()
    header = f.readline().strip()
    header = header.split(' ')
    length = int(header[0]) * int(header[1])
    img = []
    line = f.readline()

    while len(line) > 0:
        for x in line.split():
            img.append(int(x))
        line = f.readline()

    if length != len(img):
        raise Exception("Size mismatch in file between header and contents.")

    return img, int(header[0]), int(header[1])

def driveToPoint(X_GOAL, Y_GOAL):
    while(1):
        # get robots position
        positions = VWGetPosition() 
        x = positions[0]
        y = positions[1]
        angle = positions[2]

        # calculate angle to goal
        theta = math.atan2(Y_GOAL - y, X_GOAL - x)
        theta = math.degrees(theta)

        if (theta > 180.0):
            theta -= 360.0

        # find difference between current angle and goal angle
        diff = round(theta-angle)

        if (abs(diff) > 2):
            VWSetSpeed(20, diff)
        else: 
            VWSetSpeed(100, 0)
        
        # if reached goal position
        if (abs(X_GOAL - x) < 25 and abs(Y_GOAL - y) < 25):
                VWSetSpeed(0, 0)
                return 0

def heuristic(node: Node):
    endNode = nodes[len(nodes) - 1]
    return math.sqrt((node.x - endNode.x)**2 + (node.y - endNode.y)**2)

def euclidean(node: Node, node2: Node):
    return math.sqrt((node.x - node2.x)**2 + (node.y - node2.y)**2)

def getPathAStar():
    startNode = nodes[0]
    endNode = nodes[len(nodes) - 1]

    openSet = [startNode]

    closedSet = []

    parent = [0] * len(nodes)

    # cost of the cheapest path from start to n currently known
    gScore = [float('inf')] * len(nodes)
    gScore[0] = 0

    # represents our current best guess as to how cheap a path could be from start to finish if it goes through n
    fScore = [float('inf')] * len(nodes)
    fScore[0] = heuristic(startNode)

    while len(openSet) != 0:
        current = openSet[fScore.index(min(fScore))]
        if current.id == endNode.id: # at the goal
            # Retrieve path from parents and return
            currentNode = endNode
            previousNode = endNode
            # totalDistance = 0
            path = []
            while currentNode != startNode:
                path.append(currentNode)
                # totalDistance += euclidean(previousNode, currentNode)
                previousNode = currentNode
                currentNode = parent[currentNode.id - 1]
                pass

            path.append(currentNode) # the last node
            path.reverse() # put it in the right order

            # print("Path distance: ", totalDistance)
            print("Shortest path: ", [node.id for node in path])
            return path
        

        openSet.remove(current)
        closedSet.append(current)

        for edge in current.neighbours:
            
            neighbour = edge.neighbour
            if neighbour in closedSet:
                continue
            tentativeGScore = gScore[current.id - 1] + euclidean(current, neighbour)
            if tentativeGScore < gScore[neighbour.id - 1]: # tentative path is better than recorded path
                parent[neighbour.id-1] = current
                gScore[neighbour.id-1] = tentativeGScore
                fScore[neighbour.id-1] = tentativeGScore + heuristic(neighbour)
                if neighbour not in openSet:
                    openSet.append(neighbour)
    
    print("No path found")
    return

def lineSquareIntersection(Px, Py, Ax, Ay, Bx, By):
    F = (By - Ay)*Px + (Ax - Bx)*Py + (Bx*Ay - Ax*By)
    return F

def reMap(old_value, old_min, old_max, new_max, new_min):
    new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    return int(new_value)

def collisionFreePath():
    for id_1 in range(freeSquareCount):
        for id_2 in range(freeSquareCount):
            Ax = nodes[id_1].x
            Bx = nodes[id_2].x
            Ay = nodes[id_1].y
            By = nodes[id_2].y

            if id_1 == id_2:
                continue

            for square in occupiedSquares:
                Ux = square.x + square.size
                Rx = square.x
                Uy = square.y + square.size
                Ry = square.y 

                # for all pairs of free squares
                overOccupiedSquares = False

                # chech all occupied squares to see if any intesect the path between the two free squares
                negativeFs = 0
                positiveFs = 0

                if (lineSquareIntersection(Rx, Ry, Ax, Ay, Bx, By) > 0):
                    positiveFs +=1
                else:
                    negativeFs +=1

                if (lineSquareIntersection(Ux, Uy, Ax, Ay, Bx, By) > 0):
                    positiveFs +=1
                else:
                    negativeFs +=1

                if (lineSquareIntersection(Ux, Ry, Ax, Ay, Bx, By) > 0):
                    positiveFs +=1
                else:
                    negativeFs +=1

                if (lineSquareIntersection(Rx, Uy, Ax, Ay, Bx, By) > 0):
                    positiveFs +=1
                else:
                    negativeFs +=1

                if(negativeFs == 4 or positiveFs == 4):
                    # all points above or below line
                    # no intersection, check the next occupied square
                    continue
                
                else:
                    # get variables as needed for formula

                    # formula to determine if square is occupied
                    overOccupiedSquares = not ((Ax > Ux and Bx > Ux) or (Ax < Rx and Bx < Rx) or (Ay > Uy and By > Uy) or (Ay < Ry and By < Ry))

                    if (overOccupiedSquares):
                        # this is not a collision free path
                        break
                
            if (not overOccupiedSquares):
                # a collision-free path can be found, so draw it and print distance
                LCDLine(Ax+BUFFER,Ay+BUFFER,Bx+BUFFER,By+BUFFER, 0x0000FF)
                distance = math.sqrt(math.pow(Ax-Bx, 2) + math.pow(Ay - By, 2))
                LCDx = reMap((Ax+Bx)/2,0,IMAGE_SIZE,3,12)
                LCDy = reMap((Ay+By)/2,0,IMAGE_SIZE,6,28)
                LCDSetPrintf(LCDx, LCDy, str(int(distance)))
                # box rows 3->12 and columns 6->28
                print("Distance from ", Ax, Ay, " -> ", Bx, By, ": ", int(distance))

                # adding neighbours to available nodes
                neighbour = id_2
                neighbourNode = nodes[neighbour]
                neighbour_distance = math.sqrt((Ax - neighbourNode.x)**2 + (Ay - neighbourNode.y)**2)
                nodes[id_1].addNeighbour(Edge(neighbourNode, neighbour_distance))
                
def printImage(width, height, image):
    LCDImageStart(BUFFER, BUFFER, width, height)
    LCDImageSize(len(image))
    byte_array = [255 if i == 0 else 0 for i in image]
    x = (ctypes.c_byte * len(byte_array))(*byte_array)
    LCDImageBinary(x)

# main ----------------------------------------------------------------------

if __name__ == "__main__":    
    VWSetSpeed(0,0)
    SIMSetRobot(0, 200, 3800, 100, 0)
    VWSetPosition(200, 3800, 0)
    image, width, height = read_image_file(IMAGE_FILE)
    LCDMenu("Quadtree", "PATH", "DRIVE", "EXIT")

    while (1):
        key = KEYRead()
        if (key == KEY1):
            print("\nExperiment 1\n---\n")
            printImage(width, height, image)
            quadTree(0,0,IMAGE_SIZE,image)
            nodes = nodes[:freeSquareCount]
            occupiedSquares = occupiedSquares[:occupiedSquareCount]
            pass

        if (key == KEY2):
            print("\nExperiment 2\n---\n")
            collisionFreePath()
            path = getPathAStar()

        if (key == KEY3):
            print("\nExperiment 3\n---\n")
            path = getPathAStar()
            previousNode = path[0]
            for node in path:
                LCDCircle(node.x+BUFFER, node.y+BUFFER, 7, ORANGE, 1)
                LCDLine(node.x+BUFFER,node.y+BUFFER,previousNode.x+BUFFER,previousNode.y+BUFFER, ORANGE)
                driveToPoint((node.x/IMAGE_SIZE)*WORLD_SIZE,(1-(node.y/IMAGE_SIZE))*WORLD_SIZE)
                previousNode = node
            LCDLine(node.x+BUFFER,node.y+BUFFER,previousNode.x+BUFFER,previousNode.y+BUFFER, ORANGE)
            print("Path Driven")
            pass

        if (key == KEY4):
            break