#!/usr/bin/env python3
from eye import *
from math import *
from ctypes import *
from random import *

def main():
    img = []
    l=0
    f=0
    r=0
    safe=300
    LCDMenu("","","","END")
    CAMInit(QVGA)
    while(KEYRead() != KEY4):
        img = CAMGet()
        LCDImage(img)
        l=PSDGet(PSD_LEFT)
        f=PSDGet(PSD_FRONT)
        r=PSDGet(PSD_RIGHT)
        LCDSetPrintf(18,0,"PSD  L%4d  F%4d  R%4d  ",l,f,r)
        if(l>safe and f>safe and r>safe):
            VWStraight(100,200)
        else:
            VWStraight(-25,50)
            VWWait()
            dir=int(180*(random()-0.5))
            LCDSetPrintf(19,0,"Turn  %d",dir)
            VWTurn(dir,45)
            VWWait()
            LCDSetPrintf(19,0,"    ")
        OSWait(100)

main()
