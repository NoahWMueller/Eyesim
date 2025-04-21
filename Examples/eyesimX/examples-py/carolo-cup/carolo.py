#!/usr/bin/env python3
from random.eye import *

SAFE = 300
PSD_LEFT = 1
PSD_FRONT = 2
PSD_RIGHT = 3

def main():
    img = []
    LCDMenu("", "", "", "END")
    CAMInit(QVGA)
    VWStraight( 600, 200) 
    while(KEYRead() != KEY4):
        img = CAMGet() # demo
        LCDImage(img)  # only
        l = PSDGet(PSD_LEFT)
        f = PSDGet(PSD_FRONT)
        r = PSDGet(PSD_RIGHT)
        LCDSetPrintf(18,0, "PSD L%3d F%3d R%3d", l, f, r)

main()
