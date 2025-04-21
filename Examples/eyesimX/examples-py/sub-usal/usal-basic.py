#!/usr/bin/env python3
from random.eye import *
from time import sleep

#PSD IDs.
PSD_FRONT = 1
PSD_LEFT = 2
PSD_RIGHT = 3

#Thruster IDs.
HORIZONTAL = 1
VERTICAL = 2
FRONT = 3

#Fin servo IDs.
FIN_UPPER = 3
FIN_LOWER = 4
SPEED = 100

def up(secs):
    MOTORDrive(VERTICAL, SPEED)
    sleep(secs)
    MOTORDrive(VERTICAL, 0)

def down(secs):
    MOTORDrive(VERTICAL, -SPEED)
    sleep(secs)
    MOTORDrive(VERTICAL, 0)

def forwards(secs):
    MOTORDrive(HORIZONTAL, SPEED)
    sleep(secs)
    MOTORDrive(HORIZONTAL, 0)

def backwards(secs):
    MOTORDrive(HORIZONTAL, -SPEED)
    sleep(secs)
    MOTORDrive(HORIZONTAL, 0)

def left(secs):
    MOTORDrive(FRONT, SPEED)
    sleep(secs)
    MOTORDrive(FRONT, 0)

def right(secs):
    MOTORDrive(FRONT, -SPEED)
    sleep(secs)
    MOTORDrive(FRONT, 0)

def set_angle(angle):
    mappedAngle = (int)((angle + 90) * 1.41666)
    SERVOSet(FIN_UPPER, mappedAngle)
    SERVOSet(FIN_LOWER, mappedAngle)

def main():
    time = 3.0
    finAngle = 60
    set_angle(-finAngle)
    forwards(time)
    set_angle(finAngle)
    forwards(time)
    sleep(2)
    set_angle(finAngle)
    backwards(time)
    set_angle(-finAngle)
    backwards(time)


if __name__=="__main__":
    main()
