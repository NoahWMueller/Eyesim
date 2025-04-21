#!/usr/bin/env python3
from random.eye import *

def main():
    size = False
    gray = True
   
    LCDMenu("SIZE", "COL/GRAY", "", "END")
    CAMInit(QQVGA)     # automatically sets LCDIMageSize and IPSize
    VWSetSpeed(0, 90)  # rotate on the spot
    while True:
        key = KEYRead()
        if key == KEY1:
            size = not size
            if size:
                CAMInit(QVGA)
            else:
                CAMInit(QQVGA)
                LCDClear()
                LCDMenu("SIZE", "COL/GRAY", "", "END")
        elif key == KEY2:
            gray = not gray
        elif key == KEY4:
            return 0
        
        if gray:
            img = CAMGetGray()
            LCDImageGray(img)
        else: #color
            img = CAMGet()
            LCDImage(img)

main()
