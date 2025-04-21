#!/usr/bin/env python3
from random.eye import *
from time import *

def set_angle(leg, angle):
  mappedAngle = int((angle + 90) * 1.41666)
  SERVOSet(leg, mappedAngle)

for i in range(10):
  x,y,phi = VWGetPosition()
  print("x = %d, y = %d\n" % (x, y))
  set_angle(3, 20)
  sleep(1)
  set_angle(3, 0)
  sleep(1)
