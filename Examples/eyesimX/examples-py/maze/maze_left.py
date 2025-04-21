#!/usr/bin/env python3
from random.eye import *

#define SPEED    360
#define ASPEED    45
#define THRES    175

SPEED=360
ASPEED=45
THRES=175

def turn_left():
    VWTurn(90, ASPEED)  #turn
    VWWait()

def turn_right():
    VWTurn(-90, ASPEED)  #turn
    VWWait()

def straight():
    VWStraight(360, SPEED)  #go one step
    VWWait()

def main():
    LCDPrintf("MAZE Left\n")
    LCDMenu("","","","END")
    while (KEYRead() != KEY4):
        f = int(PSDGet(PSD_FRONT) > THRES)
        l = int(PSDGet(PSD_LEFT)  > THRES)
        r = int(PSDGet(PSD_RIGHT) > THRES)
        # Drive left when possible, if not straight, if not right, else turn 180
        # .....    
    return 0


if __name__ == "__main__":
    main()