#/usr/bin/env python3
from random.eye import *


def main():
    LCDMenu("","","","END")
    LCDSetColor(0xFF0000,0)
    for i in range(0,2):
        LCDPrintf("%2d RED RED RED \n",i)
        LCDSetColor(0x00FF00, 0)
    for i in range(0,2):
        LCDPrintf("%2d GREEN GREEN GREEN \n",i)
        LCDSetColor(0x0000FF, 0)
    for i in range(0,2):
        LCDPrintf("%2d BLUE BLUE BLUE \n",i)
        LCDSetColor(0xFF0000,0xFFFFFF)
    for i in range(0,2):
        LCDPrintf("%2d RED RED RED \n",i)
        LCDSetColor(0x00FF00,0xFFFFFF)
    for i in range(0,2):
        LCDPrintf("%2d GREEN GREEN GREEN\n", i)
        LCDSetColor(0x0000FF, 0xFFFFFF)
    for i in range(0,2):
        LCDPrintf("%2d BLUE BLUE BLUE\n", i)
    KEYWait(KEY4)


if __name__=="__main__":
    main()
    
