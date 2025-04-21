#!/usr/bin/env python3
from random.eye import *

LCDMenu("BEEP","","","END")
k=0;
while k!=KEY4:
  k = KEYWait(ANYKEY)
  if k==KEY1:
    AUPlay("beep.mp3")
