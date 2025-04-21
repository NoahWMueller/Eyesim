#!/usr/bin/env python
# Noah Mueller 23356892 /04/2024
from eye import *
import math

# Globals
nodes = []

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

        if (abs(diff) > 1.5):
            VWSetSpeed(0, diff)
        else: 
            VWSetSpeed(100, 0)
        
        # if reached goal position
        if (abs(X_GOAL - x) < 25 and abs(Y_GOAL - y) < 25):
                VWSetSpeed(0, 0)
                return 0

def LCDPixelBigger(x, y, colour):
    LCDArea(int(x*2),-int(y*2), int(x*2 + 2), -int(y*2 + 2), colour, 1)

def readFile(FILE_NAME):
    global nodes
    with open(FILE_NAME, "r") as f:
        lines = f.readlines()
        nodes = [None] * len(lines)

        id = 1
        for line in lines:
            entries = [int(i) for i in line.strip().split()]

            node = nodes[id-1]
            if node is None:
                node = Node(id, entries[0], entries[1])
                nodes[id - 1] = node

            neighbours = entries[2:]
            LCDCircle(2*int(128*node.x/2000), 256-2*int(128*node.y/2000), 7, WHITE, 1)

            for neighbour in neighbours:
                neighbourNode = nodes[neighbour - 1]
                if neighbourNode is None:
                    neighbourEntries = [int(x) for x in lines[neighbour - 1].strip().split()]
                    neighbourNode = Node(neighbour, neighbourEntries[0], neighbourEntries[1])
                    nodes[neighbour - 1] = neighbourNode
                
                distance = math.sqrt((node.x - neighbourNode.x)**2 + (node.y - neighbourNode.y)**2)
                node.addNeighbour(Edge(neighbourNode, distance))
                LCDLine(2*int(128*node.x/2000),256-2*int(128*node.y/2000),2*int(128*neighbourNode.x/2000),256-2*int(128*neighbourNode.y/2000), WHITE)

            id += 1

def printAdjacencyMatrix():
    for node in nodes:
        distances = [-1] * len(nodes)
        for edge in node.neighbours:
            distances[edge.neighbour.id - 1] = edge.distance

        distances[node.id - 1] = 0

        output = ''
        for distance in distances:
            output += '%.2f\t' % distance
        print(output)

def heuristic(node: Node):
    endNode = nodes[len(nodes) - 1]
    return math.sqrt((node.x - endNode.x)**2 + (node.y - endNode.y)**2) # TODO get euclidean distance

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
            totalDistance = 0
            path = []
            while currentNode != startNode:
                path.append(currentNode)
                totalDistance += euclidean(previousNode, currentNode)
                previousNode = currentNode
                currentNode = parent[currentNode.id - 1]
                pass

            path.append(currentNode) # the last node
            path.reverse() # put it in the right order

            print("Path distance: ", totalDistance)
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

if __name__ == "__main__":
    VWSetSpeed(0,0)
    SIMSetRobot(0, 300, 300, 100, -90)
    VWSetPosition(100, 100, 90)

    LCDMenu("branch", "cycle", "grid", "nodes")

    # choosing function
    while (1):
        key = KEYRead()
        if (key == KEY1):
            readFile("branch.txt")
            printAdjacencyMatrix()
            break
        if (key == KEY2):
            readFile("cycle.txt")
            printAdjacencyMatrix()
            break
        if (key == KEY3):
            readFile("grid.txt")
            printAdjacencyMatrix()
            break
        if (key == KEY4):
            readFile("nodes.txt")
            printAdjacencyMatrix()
            break

    LCDMenu("PATH","","","")
    KEYWait(KEY1)
    
    path = getPathAStar()


    LCDMenu("DRIVE","","","")
    KEYWait(KEY1)
    if path != None:
        previousNode = path[0]
        for node in path:
            LCDCircle(2*int(128*node.x/2000), 256-2*int(128*node.y/2000), 7, BLUE, 1)
            driveToPoint(node.x,node.y)
            LCDLine(2*int(128*node.x/2000),256-2*int(128*node.y/2000),2*int(128*previousNode.x/2000),256-2*int(128*previousNode.y/2000), BLUE)
            previousNode = node
        LCDLine(2*int(128*node.x/2000),256-2*int(128*node.y/2000),2*int(128*previousNode.x/2000),256-2*int(128*previousNode.y/2000), BLUE)
    else:
        print("No path to drive")
    LCDMenu("EXIT","","","")
    KEYWait(KEY1)