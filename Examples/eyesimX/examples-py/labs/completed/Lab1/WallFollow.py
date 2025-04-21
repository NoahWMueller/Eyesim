#!/usr/bin/env python3
from eye import *
SAFE = 550

#negative speed is clockwise 
#positive is counter clockwise
#sum 4878 or 5338
# motor 1 left side motor 3 right side

LCDMenu("Start","","","")
KEYWait(KEY1)

def stop_motors():
    MOTORDrive(1,0)
    MOTORDrive(3,0)

def orientate_towards_wall(input):
    while((PSDGet(PSD_LEFT)+PSDGet(PSD_RIGHT)) != input):
        print("sum = " + str(PSDGet(PSD_LEFT)+PSDGet(PSD_RIGHT)))
        MOTORDrive(1,25)
        MOTORDrive(3,-25)
    stop_motors()

def drive_to_wall():
    while (PSDGet(PSD_FRONT) > SAFE):
            MOTORDrive(1,100)
            MOTORDrive(3,100)
    stop_motors()

def turn(input):
    MOTORDrive(1,100)
    MOTORDrive(3,-100)
    OSWait(2500)
    orientate_towards_wall(input)

orientate_towards_wall(4878)
drive_to_wall()

side = 5338

for i in range(5):
    print("side = " + str(side))
    turn(side)
    drive_to_wall()
    if side == 5338: 
         side = 4878
    elif side == 4878: 
         side = 5338
        
stop_motors()
