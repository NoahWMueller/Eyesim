#!/usr/bin/env python3
from random.eye import *

LCDMenu("1","2","3","4")
LCDPrintf("KEY Get\n")
for i in range(0,4):
   LCDPrintf("press key (get) ")
   k=KEYGet()
   LCDPrintf("Key pressed is: %d\n",k)

LCDPrintf("\nKEY Read\n")
for i in range(0,2):
   LCDPrintf("press key(read)")
   k=0
   while not k:
      k=KEYRead()
   LCDPrintf("Key pressed is: %d\n",k)

LCDPrintf("\nPress END\n")
LCDMenu("","","","END")
KEYWait(KEY4)
