#!/usr/bin/env python3
from random.eye import *

def main():
    LCDMenu("","","","END")
    for i in range(0,10):
        LCDPrintf("%2d Text Text Text\n",i)
    KEYWait(KEY4)

if __name__=="__main__":
    main()
