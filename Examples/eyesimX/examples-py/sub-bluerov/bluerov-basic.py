#!/usr/bin/env python3
from random.eye import *
from time import sleep
#PSD IDs.
PSD_FRONT = 1
PSD_LEFT = 2
PSD_RIGHT = 3
PSD_DOWN = 4

#Thruster IDs.
TOP_LEFT = 1
TOP_RIGHT = 2
FRONT_LEFT = 3
FRONT_RIGHT = 4
BACK_LEFT = 5
BACK_RIGHT = 6
SPEED = 50

def up(secs):
    MOTORDrive(TOP_LEFT, SPEED)
    MOTORDrive(TOP_RIGHT, SPEED)
    sleep(secs)
    MOTORDrive(TOP_LEFT, 0)
    MOTORDrive(TOP_RIGHT, 0)

def down(secs):
    MOTORDrive(TOP_LEFT, -SPEED)
    MOTORDrive(TOP_RIGHT, -SPEED)
    sleep(secs)
    MOTORDrive(TOP_LEFT, 0)
    MOTORDrive(TOP_RIGHT, 0)


def forwards(secs):
    MOTORDrive(FRONT_LEFT, SPEED)
    MOTORDrive(FRONT_RIGHT, SPEED)
    MOTORDrive(BACK_LEFT, -SPEED)
    MOTORDrive(BACK_RIGHT, -SPEED)
    sleep(secs)
    MOTORDrive(FRONT_LEFT, 0)
    MOTORDrive(FRONT_RIGHT, 0)
    MOTORDrive(BACK_LEFT, 0)
    MOTORDrive(BACK_RIGHT, 0)

def backwards(secs):
    MOTORDrive(FRONT_LEFT, -SPEED)
    MOTORDrive(FRONT_RIGHT, -SPEED)
    MOTORDrive(BACK_LEFT, SPEED)
    MOTORDrive(BACK_RIGHT, SPEED)
    sleep(secs)
    MOTORDrive(FRONT_LEFT, 0)
    MOTORDrive(FRONT_RIGHT, 0)
    MOTORDrive(BACK_LEFT, 0)
    MOTORDrive(BACK_RIGHT, 0)

def turn_left(secs):
    MOTORDrive(FRONT_LEFT, -SPEED)
    MOTORDrive(FRONT_RIGHT, SPEED)
    MOTORDrive(BACK_LEFT, SPEED)
    MOTORDrive(BACK_RIGHT, -SPEED)
    sleep(secs)
    MOTORDrive(FRONT_LEFT, 0)
    MOTORDrive(FRONT_RIGHT, 0)
    MOTORDrive(BACK_LEFT, 0)
    MOTORDrive(BACK_RIGHT, 0)
    
def turn_right(secs):
    MOTORDrive(FRONT_LEFT, SPEED)
    MOTORDrive(FRONT_RIGHT, -SPEED)
    MOTORDrive(BACK_LEFT, -SPEED)
    MOTORDrive(BACK_RIGHT, SPEED)
    sleep(secs)
    MOTORDrive(FRONT_LEFT, 0)
    MOTORDrive(FRONT_RIGHT, 0)
    MOTORDrive(BACK_LEFT, 0)
    MOTORDrive(BACK_RIGHT, 0)

def main():
    time = 3.0
    up(time)
    forwards(time)
    turn_left(time)
    turn_right(time)
    backwards(time)
    down(time)

if __name__=="__main__":
    main()


