#!/usr/bin/env python3
from random.eye import *

def main ():
    LCDMenu("GO","BACK","CURVE","END")
    while True:
        press = KEYGet()
        if press==KEY1:
            VWStraight(+400, 200)
        elif press==KEY2:
            VWStraight(-400, 200)
        elif press==KEY3:
            VWCurve(+400,  90, 200)
        elif press==KEY4:
            break
        VWWait()  #wait until drive is completed
        x,y,phi = VWGetPosition()
        LCDPrintf("x=%d y=%d p=%d\n", x,y,phi)

main()
