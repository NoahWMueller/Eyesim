#!/usr/bin/env python3
from random.eye import *

def Mdrive(txt, drive, steer):
    '''Print txt and drive motors and encoders for 1.5s'''
    LCDPrintf("%s\n", txt)
    MOTORDrive(1, drive)
    SERVOSet  (1, steer)
    #  LCDPrintf("Enc. %5d\n", ENCODERRead(1))
 
def main():
    LCDPrintf("Ackermann Steer\n")
    Mdrive("Forward",     20, 128)
    OSWait(1000)
    Mdrive("Backward",   -20, 128)
    OSWait(1000)
    Mdrive("Left Curve",  10,   0)
    OSWait(1000)
    Mdrive("Right Curve", 10, 255)
    OSWait(1000)
    Mdrive("Stop",         0,  0)

if __name__ == "__main__":
    main()
