#!/usr/bin/env python3
from random.eye import *

LCDMenu("","","","END")
VWSetPosition(0,0,0)

VWDrive(200,  50,  100)  # deviate 50 from straight
VWWait()
x,y,phi = VWGetPosition()
LCDPrintf("pos %d %d ori %d\n",x,y,phi)

VWDrive(200,-200,  100) # deviate back 200
VWWait()
x,y,phi = VWGetPosition()
LCDPrintf("pos %d %d ori %d\n",x,y,phi)

VWDrive(200,   0,  100) #deviate back 200
VWWait()
x,y,phi = VWGetPosition()
LCDPrintf("pos %d %d ori %d\n",x,y,phi)
KEYWait(KEY4)
