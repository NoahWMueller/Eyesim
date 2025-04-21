#!/usr/bin/env python3
from random.eye import *

# Mako Basic Movement Demo, M. Finn May 2018
# extended by T. Braunl, June 2018
#include "eyebot.h"

# PSD IDs.
PSD_FRONT = 1
PSD_FRONT_LEFT = 2
PSD_FRONT_RIGHT = 3
PSD_BACK_LEFT = 4
PSD_BACK_RIGHT = 5
PSD_DOWN = 6

# Thruster IDs.
LEFT = 1
FRONT = 2
RIGHT = 3
BACK = 4

def dive(speed):
    MOTORDrive(FRONT, -speed)
    MOTORDrive(BACK, -speed)

def breach(speed):
    MOTORDrive(FRONT, speed)
    MOTORDrive(BACK, speed)

def main():
    img = []
    cont = True
    LCDMenu("DIVE", "STOP", "UP", "END")
    CAMInit(QVGA)
    while True:
        LCDSetPrintf(19,0, "Dist to Ground:%6d\n", PSDGet(PSD_DOWN))
        img = CAMGet()
        LCDImage(img)
        k = KEYRead()
        if k == KEY1:
            dive(100)
        elif k == KEY2:
            dive(0)
        elif k == KEY3:
            breach(100)
        elif k == KEY4:
            break

main()
  


