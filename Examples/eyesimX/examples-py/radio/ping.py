#!/usr/bin/env python3
from random.eye import *

# Ping-Pong radio communication program
# T. Braunl, May 2017

#include "eyebot.h"
partnerid = 0

ids = []*256

LCDMenu("MASTER", "SLAVE", "SELF", "END")
RADIOInit()
myid = RADIOGetID()
LCDPrintf("my id %d\n", myid)

k = KEYGet()
if k==KEY4: exit()
if k==KEY1 or k==KEY3: # master only
   LCDPrintf("scanning (takes time ...)\n")
   [total,ids] = RADIOStatus() 
   LCDPrintf("Total of %d robots: ", total)
   for i in range(total): LCDPrintf(" %d", ids[i])
   LCDPrintf("\n")

   # select partner robot
   if k==KEY3:  # send to self
      partnerid=myid
   else:
      if total<2: 
         LCDPrintf("Not enough partners\n") 
      partnerid = ids[0]
      if myid==partnerid: partnerid = ids[1]
   LCDPrintf("partner is %d\n", partnerid)

   LCDPrintf("I will start\n")
   RADIOSend(partnerid, '0 start')

# code for both sender and receiver
LCDPrintf("waiting ...\n")
for i in range(5):
   [partnerid, buf] = RADIOReceive()
   LCDPrintf("received from %d text: %s\n", partnerid, buf)
   count = ord(buf[0]) + 1
   RADIOSend(partnerid, chr(count) + ' reply')
KEYWait(KEY4)

