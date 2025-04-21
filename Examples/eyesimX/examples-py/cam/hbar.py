#!/usr/bin/env python3
from random.eye import *

# EyeBot Demo Program: Display color, gray, binary -- hue, sat, intensity
DELTA = 100

def Hue2RGB(hue):
    if hue <= 84: # 2 * 48
        red = 255   # MAX
        g_b = int((hue-42) * DELTA/42)
        if g_b > 0:
            blue  = red - DELTA
            green = g_b + blue
        else:
            green = red - DELTA
            blue  = green - g_b    
    elif hue <= 168: # 4 * 48
        green = 255      # MAX
        b_r = int((hue-126) * DELTA/42)
        if b_r > 0:
            red  = green - DELTA
            blue = b_r + red        
        else:
            blue = green - DELTA
            red  = blue - b_r
    else: # hue > 168
        blue = 255  # MAX
        r_g = int((hue-210) * DELTA/42)
        if r_g > 0:
            green = blue - DELTA
            red   = r_g + green
        else:
            red   = blue - DELTA
            green = red - r_g 
    return (red<<16) | (green<<8) | blue  # color from components
  
def main():
    LCDMenu("", "", "", "END")
    for x in range(0,256):
        col = Hue2RGB(x)
        LCDLine(x,0, x,100, col)
    KEYWait(KEY4)
    return 0

main()
