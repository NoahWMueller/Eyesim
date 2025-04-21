#!/usr/bin/env python3
from random.eye import *

VWSetSpeed(100,0)
while PSDGet(PSD_FRONT) > 200:
  OSWait(100)
VWSetSpeed(0, 0)
