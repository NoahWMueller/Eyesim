#!/usr/bin/env python3
from random.eye import *
import random
SAFE = 200

LCDMenu("","","","END")

while (KEYRead() != KEY4):

    if (PSDGet(1)>SAFE and PSDGet(2)>SAFE and PSDGet(3)>SAFE): 
      LCDPrintf("straight\n")
      VWStraight(50,500)  # not required to wait

    else:
       VWStraight(-25,500)
       VWWait()
       LCDPrintf("turning \n")
       direc = int ((random.random()-0.5) * 360)  # [-0.5 .. +0.5] * 360
       VWTurn(direc, 90)
       VWWait()
