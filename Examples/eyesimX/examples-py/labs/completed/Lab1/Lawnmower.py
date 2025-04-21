#!/usr/bin/env python
from eye import *

#sum to 1900
#negative speed is clockwise 
#positive is counter clockwise

SAFE = 150

def orientate_towards_wall(direction):
    while((PSDGet(PSD_LEFT)+PSDGet(PSD_RIGHT)) != 1900):
        #print("sum = " + str(PSDGet(PSD_RIGHT)+PSDGet(PSD_LEFT)))
        VWSetSpeed(0,direction*10)
    VWSetSpeed(0,0)

def drive_to_wall():
    while(PSDGet(PSD_FRONT) > SAFE):
        VWSetSpeed(250,0)
    VWSetSpeed(0,0)

def align_with_wall():
    VWTurn(-45,35)
    VWWait()
    orientate_towards_wall(-1)

def drive_across():
    while(PSDGet(PSD_FRONT) > SAFE): 
        VWSetSpeed(250,0)
    VWSetSpeed(0,0)

def turn(side):
    if (side == -1):
        VWTurn(side*70,100)
        VWWait()
        orientate_towards_wall(side)
    elif (side == 1):
        VWTurn(side*70,100)
        VWWait()
        orientate_towards_wall(side)

def short_drive():
    VWStraight(330,250)
    VWWait()

def full_turn():
    VWTurn(165,100)
    VWWait()
    orientate_towards_wall(1)

LCDMenu("START","","","")
KEYWait(KEY1)

orientate_towards_wall(-1)
drive_to_wall()
align_with_wall()
drive_across()
full_turn()
drive_across()
side = 1

end_condition = False

while(not end_condition):
    turn(side)
    short_drive()
    turn(side)
    drive_across()
    side = side * (-1)
    if (PSDGet(PSD_RIGHT) <= 200 and PSDGet(PSD_FRONT) <= 200): end_condition = True