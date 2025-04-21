#!/usr/bin/env python3
# Pixel-wise processwing in Python
# T. Braunl, April 2020
from random.eye import *
mid = 118*320 +159 # mid point of image

LCDMenu("", "", "", "END")
CAMInit(QVGA)
while (KEYRead() != KEY4):
  img = CAMGetGray()
  x = img[mid] & 0xff                          # make unsigned
  LCDSetPrintf(19,0, "image mid point: %3d ", x)
  img[mid]=255; img[mid-1]=255; img[mid+1]=255 # mark pont as white cross
  img[mid-320]=255; img[mid+320]=255
  LCDImageGray(img)
