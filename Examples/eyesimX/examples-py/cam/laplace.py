#!/usr/bin/env python3
from random.eye import *

CAMInit(QVGA)
while True:
  gray = CAMGetGray()
  edge = IPLaplace(gray)
  LCDImageGray(edge)
