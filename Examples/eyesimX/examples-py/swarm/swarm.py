#!/usr/bin/env python3
from random.eye import *
from random import *

SAFE = 300
PSD_FRONT = 1
PSD_LEFT  = 2
PSD_RIGHT = 3

def main():
    img = []
    stop = False
    id = OSMachineID() # unique ID
    # NOTE: Only 1 robot (ID==1) will use LCD or KEYs
    ME = (id==1)
    # testing
    LCDSetPrintf(20,0, "my id %d", id)
    print( "my id %d\n" % id)  # to console

    if ME: LCDMenu("", "", "", "END")
    CAMInit(QVGA)

    while not stop:
        img = CAMGet()    # demo
        if ME: LCDImage(img)  # only
        f = PSDGet(PSD_FRONT)
        l = PSDGet(PSD_LEFT)
        r = PSDGet(PSD_RIGHT)
        if ME: LCDSetPrintf(18,0, "PSD L%3d F%3d R%3d", l, f, r)
        if l>SAFE and f>SAFE and r>SAFE:
            VWStraight( 100, 200) # 100mm at 10mm/s
        else:
            VWStraight(-25, 50)   # back up
            VWWait()
            dir = int(((random() - 0.5))*180)
            LCDSetPrintf(19,0, "Turn %d", dir)
            VWTurn(dir, 45)      # turn random angle
            VWWait()
            if ME: LCDSetPrintf(19,0, "          ")
        OSWait(100)
        if ME: stop = (KEYRead() == KEY4)

main()
