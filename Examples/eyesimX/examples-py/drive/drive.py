#!/usr/bin/env python3
from random.eye import *

def main():
    for i in range(0,2):
        VWStraight(500, 50)#0.5m in ca. 5s
        while (not VWDone()):
            x,y,p = VWGetPosition()
            LCDPrintf("X:%4d; Y:%4d; P:%4d\n", x, y, p)
            if (PSDGet(2) < 100):
                VWSetSpeed(0,0)#STOP if obstacle in front
        VWTurn(180, 60)   #half turn (180 deg) in ca. 3s
        VWWait()         #wait until completed

main()
