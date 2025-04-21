#!/usr/bin/env python3
from random.eye import *

def Threshold(gray):
  for i in range(0, QVGA_PIXELS):
    if (gray[i] > 50):
      gray[i]=255
    else:
      gray[i]=0


LCDMenu("EDGE","OVERLAY","","END")
k    = 0
over = 1
CAMInit(QVGA)
VWSetSpeed(0,45)  # continuous turning

while k != KEY4:
  gray = CAMGetGray()
  edge = IPLaplace(gray)
  Threshold(edge)
  k = KEYRead()
  if k==KEY1: over=0
  if k==KEY2: over=1
  if over:
    col = IPOverlayGray(gray, edge, RED)
    LCDImage(col)
  else: LCDImageGray(edge)
