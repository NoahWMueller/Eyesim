#!/usr/bin/env python
from eye import *
import math
from ctypes import *

# goal positions for single spline drive
X_GOAL = 300
Y_GOAL = 1000
ANGLE_GOAL = 0

def P(u, pk, pk1, dpk, dpk1):
    H0 = 2*u**3 - 3*u**2 + 1
    H1 = -2*u**3 + 3*u**2
    H2 = u**3 - 2*u**2 + u
    H3 = u**3 - u**2
    P = H0*pk + H1*pk1 + H2*dpk + H3*dpk1
    return P

def SplineDrive(x,y,alpha):
    LCDClear()

    # scaling factor
    k = 1.5

    # defining values
    ax = START_X
    ay = START_Y
    startingAngle = START_ANGLE
    VWSetPosition(ax, ay, startingAngle)

    # estimated length of travel
    length = k*math.sqrt((x-ax)**2 + (y-ay)**2)

    # other values
    Dax = length
    Day = 0 

    # destination x and y values
    bx = x
    by = y

    # calculating Dbx and Dby
    Dbx = length*(math.cos(math.radians(alpha)))
    Dby = length*(math.sin(math.radians(alpha)))

    # lists for travel points
    x_list = []
    y_list = []

    # displaying path on lcd
    LCDCircle(int(x*(450/2000)),int(275-(y*(275/2000))),15,0x0000FF, 1) # finish

    LCDCircle(int(ax*(450/2000)),int(275-(ay*(275/2000))),15,0xFFFFFF, 1) # start

    # calculating spline points and placing them into list
    for u in range(0,11):
        u = u/10
        sp_x = P(u, ax, bx, Dax, Dbx)
        sp_y = P(u, ay, by, Day, Dby)
        LCDCircle(int(sp_x*(450/2000)),int(275-(sp_y*(275/2000))),10,0xFF0000, 1)
        x_list.append(sp_x)
        y_list.append(sp_y)

    # calculating number of steps
    steps = len(x_list)
    
    # defining starting values
    currentX = ax
    currentY = ay
    currentAngle = startingAngle

    for n in range(1,steps):
    
        # iterating through lists for values
        sp_x = x_list[n]
        sp_y = y_list[n]

        # moving
        while (currentX not in range(int(sp_x-5),int(sp_x+5)) or currentY not in range(int(sp_y-5),int(sp_y+5))):
            # get positions
            positionList = VWGetPosition()
            currentX = positionList[0]
            currentY = positionList[1]
            currentAngle = positionList[2]
            
            # calculate angle between current and desired position
            sphi = math.atan2(sp_y - currentY, sp_x - currentX)
            
            # rads to degrees
            sphi = math.degrees(sphi)

            # difference to current orientation
            rotation = sphi - currentAngle

            # rotating and driving
            if rotation not in range (int(-1),int(1)): 
                VWSetSpeed(10,int(rotation))
            else:
                VWSetSpeed(20,0)

        VWSetSpeed(0,0)

def task2():

    # getting location points from way.txt and placing them into a list
    points = []
    with open('way.txt', 'r') as file:
        for line in file.readlines():
            points.append([int(c) for c in line.strip().split()])

    # going through the list of points and calling spline drive on each
    i = 0
    while(True):
        if i == len(points):
            i = 0
        
        # obtaining positions
        positionList = VWGetPosition()

        # calculate angle between current and desired position
        sphi = math.atan2(points[i][1] - positionList[1], points[i][0] - positionList[0])

        # rads to degrees
        sphi = math.degrees(sphi)

        # defining and setting positions
        positionList = VWGetPosition()
        global START_ANGLE
        START_ANGLE = positionList[2]
        desired_angle = sphi

        # using spline drive function on desired points
        SplineDrive(points[i][0],points[i][1],desired_angle)
        if i != len(points):
            global START_X
            global START_Y
            START_X = points[i][0]
            START_Y = points[i][1]
            i+=1


def main():

    # setting start values
    global START_X
    global START_Y
    global START_ANGLE
    START_X = 500
    START_Y = 500
    START_ANGLE = 0 

    # setting robots position and movement
    SIMSetRobot(0, START_X, START_Y, 0, START_ANGLE)
    VWSetSpeed(0,0)

    # lcdmenu to choose which function to use
    LCDMenu("SplineDrive", "Task 2", "", "Abort")
    while (1):
        key = KEYRead()
        if (key == KEY1):
            SplineDrive(X_GOAL, Y_GOAL, ANGLE_GOAL)
            break
        if (key == KEY2):
            task2()
            break
        if (key == KEY4):
            return
    return

main()