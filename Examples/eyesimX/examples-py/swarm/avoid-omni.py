#!/usr/bin/env python3
from random.eye import *

S = 30  # speed [0..100]
T = 50  #distance threshold [mm] 

# Global speed variables
Fleft = 0
Fright = 0
Bleft = 0
Bright = 0

def Mdrive():
    MOTORDrive(1, Fleft)
    MOTORDrive(2, Fright)
    MOTORDrive(3, Bleft)
    MOTORDrive(4, Bright)

def setspeed(a, b, c, d):
    global Fleft, Fright, Bleft, Bright
    Fleft = a 
    Fright = b
    Bleft = c 
    Bright = d

def addspeed(a, b, c, d):
    global Fleft, Fright, Bleft, Bright
    Fleft += a 
    print("a",Fleft)
    Fright += b
    Bleft += c 
    Bright += d

# MAIN
while True:
   left  = PSDGet(1)
   front = PSDGet(2)
   right = PSDGet(3)
   back  = PSDGet(4)
   LCDPrintf("L%d F%d  R%d  B%d\n", left, front, right, back)

   Xdiff = front - back
   Ydiff = left  - right
   setspeed(0, 0, 0, 0)
   if Xdiff >  T: addspeed( S,  S,  S,  S)
   if Xdiff < -T: addspeed(-S, -S, -S, -S)
   if Ydiff >  T: addspeed(-S,  S,  S, -S)
   if Xdiff < -T: addspeed( S, -S, -S,  S)
   Mdrive()
