#!/usr/bin/env python3
from random.eye import *

while True:
  f = PSDGet(2)
  b = PSDGet(4)
  #LCDSetPrintf(0,0, "Dist. F %4d –– Dist. B %4d", f, b)
  LCDPrintf("Dist. F %3d -- B %3d", f, b)
  
  diff = int((f - b)/2)
  if abs(diff) > 10:
      VWStraight (diff, 100)
  
