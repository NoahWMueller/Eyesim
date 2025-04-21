#!/usr/bin/env python3
from random.eye import *

def Mdrive(txt, left, right):
    LCDPrintf("Distance %4d  Direction %s\n", PSDGet(1), txt)
    MOTORDrive(1,left)
    MOTORDrive(2,right)
    OSWait(1500) # Drive for 1.5 sec
    LCDPrintf("Encoder %5d %5d\n", ENCODERRead(1), ENCODERRead(2))

def main():
    Mdrive("Forward",     60, 60)
    Mdrive("Backward",   -60,-60)
    Mdrive("Left Curve",  20, 60)
    Mdrive("Right Curve", 60, 20)
    Mdrive("Turn Spot L",-20, 20)
    Mdrive("Turn Spot R", 20,-20)
    Mdrive("Stop",         0,  0)

if __name__=="__main__":
    main()
