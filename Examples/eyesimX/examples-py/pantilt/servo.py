#!/usr/bin/env python3
from random.eye import *

def checkpos(pan, tilt):
    SERVOSet(1, pan)
    SERVOSet(2, tilt)
    LCDSetPrintf(0,0,"PanTlt: %3d %3d\n", pan, tilt)
    img = CAMGet()
    LCDImage(img)

def main():
    CAMInit(QVGA)
    for pos in range(128,255): checkpos(pos, 128)
    for pos in range(255,0,-1): checkpos(pos, 128)
    for pos in range(0,128): checkpos(pos, 128)
    
    for pos in range(128,255): checkpos(128, pos)
    for pos in range(255,0,-1): checkpos(128, pos)
    for pos in range(0,128): checkpos(128, pos)


main()
