#!/usr/bin/env python
from eye import *
import math

X_GOAL = 2700 #2700
Y_GOAL = 2600 #2600
X_START = 500
Y_START = 500
ANGLE_START = 0

hit_x = 0
hit_y = 0
def get_angle():

    # get robots position
    positions = VWGetPosition() 
    x = positions[0]
    y = positions[1]
    angle = positions[2]

    # calculate angle to goal
    theta = math.atan2(Y_GOAL - y, X_GOAL - x)
    theta = math.degrees(theta)
    
    LCDSetPrintf(0, 0, "x = %d y = %d angle = %d theta = %d", x, y, angle,theta)

    if (theta > 180.0):
        theta -= 360.0

    # find difference between current angle and goal angle
    diff = round(theta-angle)

    return angle, diff, theta

def find_minimum():
    minimum = 1000
    cont = True
    while(cont):
        VWSetSpeed(0,50)
        if PSDGet(PSD_RIGHT) < minimum:
            minimum = PSDGet(PSD_RIGHT)
        else:
            VWSetSpeed(0,0)
            cont = False
            return

def Distbug(x,y):
    # setting start position for robot sensors
    ax = X_START
    ay = Y_START
    startingAngle = ANGLE_START
    VWSetPosition(ax, ay, startingAngle)

    while (1):
        # getting positional values for end condition
        positions = VWGetPosition() 
        x = positions[0]
        y = positions[1]

        # defining movement states
        DRIVING = 1
        ROTATING = 2
        FOLLOWING = 3

        # setting start state
        state = DRIVING

        # drive towards the goal
        while state == DRIVING: 
            LCDSetPrintf(5,0,"Driving")

            angle, diff, theta = get_angle()

            LCDSetPrintf(1,0,"angle = %d diff = %d theta = %d", angle, diff, theta)
            distances = LIDARGet() 

            if (distances[180]<300 or distances[30]<200 or distances[210]<200):
                VWSetSpeed(0, 0)

                # setting hit point
                positions = VWGetPosition() 
                global hit_x
                global hit_y
                hit_x = positions[0]
                hit_y = positions[1]

                # change state
                state = ROTATING

            elif (abs(diff) > 1.5):
                VWSetSpeed(20, diff)
            else: 
                VWSetSpeed(100, 0)
            
            # end condition
            positions = VWGetPosition() 
            x = positions[0]
            y = positions[1]
            if (abs(X_GOAL - x) < 50 and abs(Y_GOAL - y) < 50):
                LCDSetPrintf(3,0, "Goal found")
                VWSetSpeed(0, 0)
                return 0
        LCDClear()

        # rotate perpendicular to obstacle
        while state == ROTATING:
            LCDSetPrintf(5,0,"Rotating")
            angle, diff, theta = get_angle()
            while (PSDGet(PSD_RIGHT) > 400):
                VWSetSpeed(0,50)
            VWSetSpeed(0,0)
            find_minimum()
            state = FOLLOWING
        LCDClear()

        # follow along obstacle boundary
        while state == FOLLOWING: 
            LCDSetPrintf(5,0,"Following")
            counter = 1
            cont = True
            distance = PSDGet(PSD_RIGHT)
            LCDSetPrintf(6,0,"hit_x = %d hit_y = %d", hit_x, hit_y)
            d_min = math.sqrt((X_GOAL-hit_x)**2+(Y_GOAL-hit_y)**2)
            while(cont):
                if (PSDGet(PSD_FRONT)<distance):
                    VWTurn(90,60)
                    VWWait()
                if (PSDGet(PSD_RIGHT)>distance):
                    VWSetSpeed(50,-50*int(PSDGet(PSD_RIGHT)/distance))
                elif (PSDGet(PSD_RIGHT)<distance):
                    VWSetSpeed(50,50*int(distance/PSDGet(PSD_RIGHT)))
                else:
                    VWSetSpeed(50,0)
                
                angle, diff, theta = get_angle()
                positions = VWGetPosition() 
                distances = LIDARGet() 
                x = positions[0]
                y = positions[1]

                if (counter>100 and abs(hit_x-x)<50 and abs(hit_y-y)<50):
                    VWSetSpeed(0, 0)
                    LCDSetPrintf(3,0, "Goal unreachable")
                    return 1 # finish with error
                
                # distance differences and length
                dx = X_GOAL - x
                dy = Y_GOAL - y
                d = math.sqrt(dx*dx + dy*dy)

                # update minimum distance
                if (d < d_min):
                    d_min = d

                # calculate free space towards goal
                new_angle = 180 - (theta-angle)
                if (new_angle<0):
                    new_angle +=360
                free_distance = distances[round(new_angle)]

                LCDSetPrintf(2,0,"new_angle = %d d = %f free distance = %d d_min = %f", new_angle,d,free_distance,d_min)

                # check leave condition
                STEP = 1000
                LCDSetPrintf(7,0," d - f = %d d_min - step = %d", (d - free_distance), (d_min - STEP))
                if (d - free_distance <= d_min - STEP):
                    LCDSetPrintf(3,0, "Leaving obstacle ")
                    VWSetSpeed(0, 0)
                    VWStraight(300, 100)
                    VWWait()
                    state = DRIVING
                    cont = False

                # incrementing counter
                counter += 1
        LCDClear()



def main():
    # setting robots position and angle
    SIMSetRobot(0, X_START, Y_START, 0, ANGLE_START)
    VWSetSpeed(0,0)

    LCDMenu("DistBug", "", "", "End")

    # choosing function
    while (1):
        key = KEYRead()
        if (key == KEY1):
            Distbug(X_GOAL, Y_GOAL)
            break
        if (key == KEY4):
            break
    return

main()