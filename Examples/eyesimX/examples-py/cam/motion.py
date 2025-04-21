#!/usr/bin/env python3
# EyeBot Demo Program: Image Motion, T. Braunl, June 2015
from random.eye import *
from ctypes import *

def image_diff(i1, i2):
    diff = (c_byte * QVGA_PIXELS)()
    for i in range(QVGA_PIXELS):
        diff[i] = abs(i1[i] - i2[i])
    return diff

def avg(d):
    sum=0
    for i in range(QVGA_PIXELS):
        sum += d[i]
    return int(sum/QVGA_PIXELS)

def main():
    CAMInit(QVGA)
    LCDMenu("", "", "", "END")
    delay = 0
  
    while (KEYRead() != KEY4):
        image1 = CAMGetGray()
        OSWait(100)  # Wait 0.1s
        image2 = CAMGetGray()
        
        diff = image_diff(image1, image2)
        LCDImageGray(diff)
        avg_diff = avg(diff)
        LCDSetPrintf(0,50, "Avg = %3d", avg_diff)

        if (avg_diff > 15):  # Alarm threshold
            LCDSetPrintf(2,50, "ALARM!!!")
            delay = 10 
        if (delay):
            delay -= 1
        else:
            LCDSetPrintf(2,50, "        ") # clear text after delay
            
        VWTurn(180, 45) # keep turning slowly
        OSWait(100)     # Wait 0.1s

main()
