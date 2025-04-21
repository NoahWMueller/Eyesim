#!/usr/bin/env python3
from random.eye import *

def main():
    while True:
        f = PSDGet(2)
        b = PSDGet(4)
        LCDSetPrintf(0,0, "F%4d -- B%4d  ", f, b)
        diff = int((f-b)/2)
        if abs(diff) > 10:
             VWStraight(diff, 200)

if __name__=="__main__":
    main()

