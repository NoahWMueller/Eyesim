#!/usr/bin/env python3
from random.eye import *
from time import sleep
PSD_FRONT = 1
PSD_FRONT_LEFT = 2
PSD_FRONT_RIGHT = 3
PSD_BACK_LEFT = 4
PSD_BACK_RIGHT = 5
PSD_DOWN = 6

#Thruster IDs.
LEFT = 1
FRONT = 2
RIGHT = 3
BACK = 4

SPEED = 100

def up(secs):
    MOTORDrive(FRONT, SPEED)
    MOTORDrive(BACK, SPEED)
    sleep(secs)
    MOTORDrive(FRONT, 0)
    MOTORDrive(BACK, 0)

def down(secs):
    MOTORDrive(FRONT, -SPEED)
    MOTORDrive(BACK, -SPEED)
    sleep(secs)
    MOTORDrive(FRONT, 0)
    MOTORDrive(BACK, 0)


def forwards(secs):
    MOTORDrive(LEFT, SPEED)
    MOTORDrive(RIGHT, SPEED)
    sleep(secs)
    MOTORDrive(LEFT, 0)
    MOTORDrive(RIGHT, 0)

def backwards(secs):
    MOTORDrive(LEFT, -SPEED)
    MOTORDrive(RIGHT, -SPEED)
    sleep(secs)
    MOTORDrive(LEFT, 0)
    MOTORDrive(RIGHT, 0)

def turn_left(secs):
    MOTORDrive(LEFT, -SPEED)
    MOTORDrive(RIGHT, SPEED)
    sleep(secs)
    MOTORDrive(LEFT, 0)
    MOTORDrive(RIGHT, 0)

def turn_right(secs):
    MOTORDrive(LEFT, SPEED)
    MOTORDrive(RIGHT, -SPEED)
    sleep(secs)
    MOTORDrive(LEFT, 0)
    MOTORDrive(RIGHT, 0)

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
