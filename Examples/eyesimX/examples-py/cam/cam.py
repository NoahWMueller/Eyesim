#!/usr/bin/env python3
from eye import *

LCDMenu("", "", "", "END")
CAMInit(QVGA)
VWSetSpeed(0, 90)  # rotate on spot
while (KEYRead() != KEY4):
  img = CAMGet()
  LCDImage(img)
