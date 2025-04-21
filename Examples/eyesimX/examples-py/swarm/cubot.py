#!/usr/bin/env python3
from random.eye import *

def Mdrive(txt, left, right):
    ''' Print txt and drive motors and encoders for 1.5s '''
    LCDPrintf("%s\n", txt)
    MOTORDrive(1,left)
    MOTORDrive(2,right)
    OSWait(3500)

def main():
    LCDPrintf("Diff.Steering\n")

    Mdrive("Forward",     60, 60)
    Mdrive("Backward",   -60,-60)
    Mdrive("Left Curve",  20, 60)
    Mdrive("Right Curve", 60, 20)
    Mdrive("Turn Spot L",-20, 20)
    Mdrive("Turn Spot R", 20,-20)
    Mdrive("Stop",         0,  0)

if __name__ == "__main__":
    main()
