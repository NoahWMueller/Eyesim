#!/usr/bin/env python3
from random.eye import *
from random import *

SAFE = 300
PSD_LEFT = 1
PSD_FRONT = 2
PSD_RIGHT = 3

def main():
    LCDMenu("", "", "", "END")
    CAMInit(QVGA)
    while(KEYRead() != KEY4):
        img = CAMGet()
        LCDImage(img)
        l = PSDGet(PSD_LEFT)
        f = PSDGet(PSD_FRONT)
        r = PSDGet(PSD_RIGHT)
        LCDSetPrintf(18,0, "PSD L%3d F%3d R%3d", l, f, r)
        if (l>SAFE and f>SAFE and r>SAFE):
            VWStraight( 100, 200) # 100mm at 10mm/s
        else:
            VWStraight(-25, 50)   # back up
            VWWait()
            dir = int(((random() - 0.5))*180)
            LCDSetPrintf(19,0, "Turn %d", dir)
            VWTurn(dir, 45)      # turn random angle
            VWWait()
            LCDSetPrintf(19,0, "          ")
        OSWait(100)
    return 0

main()
