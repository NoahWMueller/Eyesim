#!/usr/bin/env python3
from random.eye import *

ids = []*256

LCDMenu("", "", "", "END")
LCDPrintf("init\n")
RADIOInit()

myid = RADIOGetID()
LCDPrintf("my id %d\n", myid)
LCDPrintf("scanning\n")
[total,ids] = RADIOStatus()

LCDPrintf("Other robots: %d\n", total)
for i in range(total):
   LCDPrintf("%d  ", ids[i])
   LCDPrintf("\n")
KEYWait(KEY4)
