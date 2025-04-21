#!/usr/bin/env python3
# updated by Alex Z, 7/10/2018
from random.eye import *

functions = 9

fname=["Forward",  "Backward", "Rotate Left", "Rotate Right",
     "Curve Left\n(FORWARD)", "Curve Right\n(FORWARD)", "Curve Left\n(BACKWARD)",
     "Curve Right\n(BACKWARD)", "SetPos [0,0,0]"]

#velocities
vel =[ [ 300,  0], [-300,   0], [   0, 30], [0, -30],
      [ 300, 30], [ 300, -30], [-300, 30], 
      [-300,-30], [ 0,  0] ]

def main():
    fnum = 0
    done = 0
    v = 0
    w = 0
    while True:
        LCDClear()
        LCDMenu("+", "-", "GO", "END")
        LCDPrintf("%s\n" % fname[fnum])
        x,y,phi = VWGetPosition()
        LCDPrintf("x = %d   \n", x)
        LCDPrintf("y = %d   \n", y)
        LCDPrintf("p = %d   \n", phi)

        v = vel[fnum][0]
        w = vel[fnum][1]
        LCDPrintf("v=%d, w=%d\n", v, w)
        k = KEYGet()
        if k == KEY1:
            fnum = (fnum+1) % functions
        elif k == KEY2:
            fnum = (fnum-1 +functions) % functions
        elif k == KEY3:
            if fnum<8:
                VWSetSpeed(v,w)
                LCDMenu(" ", " ", "STOP", " ")
                KEYWait(KEY3) # continue until key pressed
            elif fnum == 8:
                VWSetPosition(0,0,0)
        elif k == KEY4:
            done = 1
        VWSetSpeed(0,0) # stop
        if done: break

main()
